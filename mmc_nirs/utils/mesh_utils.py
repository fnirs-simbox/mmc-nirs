"""Shared validation and geometry helpers for tetrahedral meshes."""

import itertools
from collections.abc import Mapping
from typing import Literal

import numpy as np
import trimesh
from numpy.typing import ArrayLike

from .prepared_input_io import require_fields

PREPARED_MESH_KEYS = {"nodes", "elements", "element_tissue_values"}


def as_coordinate_array(values: ArrayLike, name: str) -> np.ndarray:
    """Return a finite, copied ``(n, 3)`` floating-point coordinate array."""
    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] == 0 or coordinates.shape[1] != 3:
        raise ValueError(f"{name} must be a non-empty array with shape (n, 3)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} must contain only finite values")
    return coordinates.copy()


def as_element_array(
    values: ArrayLike,
    number_of_nodes: int,
    name: str = "mesh_elements",
    *,
    allow_extra_columns: bool = True,
    index_base: Literal["auto", "zero", "one"] = "auto",
) -> np.ndarray:
    """Return zero-based tetrahedral vertex indices after validation."""
    elements = np.asarray(values)
    valid_column_count = elements.ndim == 2 and (
        elements.shape[1] >= 4 if allow_extra_columns else elements.shape[1] == 4
    )
    if elements.ndim != 2 or elements.shape[0] == 0 or not valid_column_count:
        expected_shape = "at least four columns" if allow_extra_columns else "shape (n_elements, 4)"
        raise ValueError(f"{name} must be a non-empty array with {expected_shape}")

    elements = elements[:, :4]
    if not np.issubdtype(elements.dtype, np.integer):
        if not np.all(np.isfinite(elements)) or not np.all(elements == np.floor(elements)):
            raise ValueError(f"{name} must contain integer vertex indices")
    elements = elements.astype(np.intp, copy=True)

    if index_base not in {"auto", "zero", "one"}:
        raise ValueError("index_base must be 'auto', 'zero', or 'one'")
    if index_base == "one" or (index_base == "auto" and elements.min() >= 1 and elements.max() <= number_of_nodes):
        elements -= 1
    if elements.min() < 0 or elements.max() >= number_of_nodes:
        raise ValueError(f"{name} contains an out-of-range vertex index")
    return elements


def as_element_tissue_array(
    values: ArrayLike,
    number_of_elements: int,
    name: str = "element_tissue_values",
) -> np.ndarray:
    """Return one validated integer tissue label per mesh element."""
    tissues = np.asarray(values)
    if tissues.shape != (number_of_elements,):
        raise ValueError(f"{name} must contain one value per element")
    if not np.issubdtype(tissues.dtype, np.integer):
        if not np.all(np.isfinite(tissues)) or not np.all(tissues == np.floor(tissues)):
            raise ValueError(f"{name} must contain integer labels")
    return tissues.astype(np.intp, copy=True)


def validate_prepared_mesh(prepared_mesh: Mapping[str, ArrayLike]) -> dict[str, np.ndarray]:
    """Validate and copy the canonical arrays in a prepared mesh mapping."""
    if not isinstance(prepared_mesh, Mapping):
        raise TypeError("prepared_mesh must be a mapping")
    require_fields(prepared_mesh, PREPARED_MESH_KEYS, "prepared_mesh")

    nodes = as_coordinate_array(prepared_mesh["nodes"], "prepared_mesh['nodes']")
    elements = as_element_array(
        prepared_mesh["elements"],
        len(nodes),
        "prepared_mesh['elements']",
        allow_extra_columns=False,
        index_base="zero",
    )
    tissue_values = as_element_tissue_array(
        prepared_mesh["element_tissue_values"],
        len(elements),
        "prepared_mesh['element_tissue_values']",
    )
    return {
        "nodes": nodes,
        "elements": elements,
        "element_tissue_values": tissue_values,
    }


def validate_tissue_property_coverage(
    element_tissue_values: ArrayLike,
    number_of_media: int,
) -> None:
    """Require every element label to reference a non-background MMC medium."""
    tissue_values = np.asarray(element_tissue_values)
    if tissue_values.size == 0 or tissue_values.min() < 1 or tissue_values.max() >= number_of_media:
        raise ValueError(
            "prepared_mesh['element_tissue_values'] contains labels not represented by the optical properties"
        )


def make_orientation_matrices() -> dict[str, np.ndarray]:
    """Build transforms from every valid anatomical orientation to RAS."""
    axis_letters = (("R", "L"), ("A", "P"), ("S", "I"))
    axis_vectors = {
        "R": np.array([1.0, 0.0, 0.0]),
        "L": np.array([-1.0, 0.0, 0.0]),
        "A": np.array([0.0, 1.0, 0.0]),
        "P": np.array([0.0, -1.0, 0.0]),
        "S": np.array([0.0, 0.0, 1.0]),
        "I": np.array([0.0, 0.0, -1.0]),
    }

    matrices: dict[str, np.ndarray] = {}
    for axis_order in itertools.permutations(range(3)):
        for signs in itertools.product(range(2), repeat=3):
            code = "".join(axis_letters[axis][sign] for axis, sign in zip(axis_order, signs, strict=True))
            matrices[code] = np.column_stack([axis_vectors[letter] for letter in code])
    return matrices


def _find_containing_elements(
    points: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Return the containing tetrahedron index for every point, or ``-1``."""
    tetrahedra = nodes[elements]
    origins = tetrahedra[:, 0]
    edge_matrices = np.stack(
        (
            tetrahedra[:, 1] - origins,
            tetrahedra[:, 2] - origins,
            tetrahedra[:, 3] - origins,
        ),
        axis=-1,
    )
    determinants = np.linalg.det(edge_matrices)
    valid_elements = np.flatnonzero(np.abs(determinants) > np.finfo(float).eps)
    inverse_edges = np.linalg.inv(edge_matrices[valid_elements])

    containing_elements = np.full(points.shape[0], -1, dtype=np.intp)
    for point_index, point in enumerate(points):
        local_coordinates = np.einsum(
            "eij,ej->ei",
            inverse_edges,
            point - origins[valid_elements],
        )
        barycentric_coordinates = np.column_stack((1.0 - local_coordinates.sum(axis=1), local_coordinates))
        is_inside = np.all(barycentric_coordinates >= -tolerance, axis=1) & np.all(
            barycentric_coordinates <= 1.0 + tolerance,
            axis=1,
        )
        if np.any(is_inside):
            containing_elements[point_index] = valid_elements[np.flatnonzero(is_inside)[0]]
    return containing_elements


def make_surface_mesh(nodes: np.ndarray, elements: np.ndarray) -> trimesh.Trimesh:
    """Construct the exterior triangular surface of a tetrahedral mesh."""
    tetrahedron_faces = elements[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
    _, unique_indices, face_counts = np.unique(
        np.sort(tetrahedron_faces, axis=1),
        axis=0,
        return_index=True,
        return_counts=True,
    )
    boundary_faces = tetrahedron_faces[unique_indices[face_counts == 1]]
    return trimesh.Trimesh(vertices=nodes, faces=boundary_faces, process=False)


def find_closest_nodes(nodes: ArrayLike, target_points: ArrayLike) -> np.ndarray:
    """Return the nearest mesh-node index for every three-dimensional target."""
    node_array = as_coordinate_array(nodes, "nodes")
    targets = as_coordinate_array(target_points, "target_points")
    squared_distances = np.sum((targets[:, None, :] - node_array[None, :, :]) ** 2, axis=2)
    return np.argmin(squared_distances, axis=1).astype(np.intp, copy=False)
