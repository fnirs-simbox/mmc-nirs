"""Shared validation and geometry helpers for tetrahedral meshes."""

import itertools
from collections.abc import Mapping
from numbers import Integral
from typing import Any, Literal

import numpy as np
import trimesh
from numpy.typing import ArrayLike

from .prepared_input_io import require_config_section, require_fields

PREPARED_MESH_KEYS = {
    "nodes",
    "elements",
    "element_tissue_ids",
    "ordered_tissue_ids",
    "ordered_tissues",
}
_MESH_SETTINGS_KEYS = {
    "ordered_tissues",
    "mesh_orientation",
    "mesh_units",
}


def ordered_tissue_arrays(ordered_tissues: Mapping[str | int, str]) -> tuple[np.ndarray, np.ndarray]:
    """Return MMC tissue IDs and names sorted by validated numeric ID."""
    if not isinstance(ordered_tissues, Mapping):
        raise TypeError("ordered_tissues must be a mapping from tissue IDs to tissue names")
    if not ordered_tissues:
        raise ValueError("ordered_tissues must not be empty")

    tissues_by_id: dict[int, str] = {}
    for configured_id, tissue_name in ordered_tissues.items():
        if isinstance(configured_id, bool):
            raise ValueError("ordered_tissues keys must be canonical non-negative integer strings")
        if isinstance(configured_id, Integral):
            tissue_id = int(configured_id)
        elif isinstance(configured_id, str) and configured_id.isdecimal():
            tissue_id = int(configured_id)
            if configured_id != str(tissue_id):
                raise ValueError("ordered_tissues keys must be canonical non-negative integer strings")
        else:
            raise ValueError("ordered_tissues keys must be canonical non-negative integer strings")
        if tissue_id < 0:
            raise ValueError("ordered_tissues keys must be non-negative")
        if tissue_id in tissues_by_id:
            raise ValueError(f"ordered_tissues contains duplicate tissue ID {tissue_id}")
        if not isinstance(tissue_name, str) or not tissue_name:
            raise ValueError("ordered_tissues must contain non-empty tissue names")
        tissues_by_id[tissue_id] = tissue_name

    expected_ids = set(range(len(tissues_by_id)))
    if set(tissues_by_id) != expected_ids:
        raise ValueError("ordered_tissues IDs must be contiguous from 0 to num_tissues - 1")
    tissue_names = [tissues_by_id[tissue_id] for tissue_id in range(len(tissues_by_id))]
    if len(set(tissue_names)) != len(tissue_names):
        raise ValueError("ordered_tissues must not contain duplicate tissue names")
    return np.arange(len(tissue_names), dtype=np.intp), np.asarray(tissue_names, dtype=np.str_)


def _ordered_tissue_mapping_from_arrays(
    ordered_tissue_ids: ArrayLike,
    ordered_tissues: ArrayLike,
) -> dict[int, str]:
    """Reconstruct an ID-to-name mapping from prepared-mesh arrays."""
    tissue_ids = np.asarray(ordered_tissue_ids)
    if tissue_ids.ndim != 1 or tissue_ids.size == 0:
        raise ValueError("prepared_mesh['ordered_tissue_ids'] must be a non-empty one-dimensional array")
    if not np.issubdtype(tissue_ids.dtype, np.integer):
        if not np.all(np.isfinite(tissue_ids)) or not np.all(tissue_ids == np.floor(tissue_ids)):
            raise ValueError("prepared_mesh['ordered_tissue_ids'] must contain integer IDs")
    tissue_ids = tissue_ids.astype(np.intp, copy=False)
    if len(np.unique(tissue_ids)) != len(tissue_ids):
        raise ValueError("prepared_mesh['ordered_tissue_ids'] must not contain duplicate IDs")

    if isinstance(ordered_tissues, (str, bytes)):
        raise ValueError("prepared_mesh['ordered_tissues'] must be a one-dimensional array")
    tissue_names = list(ordered_tissues)
    if len(tissue_names) != len(tissue_ids):
        raise ValueError("prepared_mesh ordered tissue IDs and names must have equal lengths")
    return dict(zip(tissue_ids.tolist(), tissue_names, strict=True))


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


def as_element_tissue_id_array(
    values: ArrayLike,
    number_of_elements: int,
    name: str = "element_tissue_ids",
) -> np.ndarray:
    """Return one validated integer MMC medium ID per mesh element."""
    tissues = np.asarray(values)
    if tissues.shape != (number_of_elements,):
        raise ValueError(f"{name} must contain one value per element")
    if not np.issubdtype(tissues.dtype, np.integer):
        if not np.all(np.isfinite(tissues)) or not np.all(tissues == np.floor(tissues)):
            raise ValueError(f"{name} must contain integer IDs")
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
    tissue_ids = as_element_tissue_id_array(
        prepared_mesh["element_tissue_ids"],
        len(elements),
        "prepared_mesh['element_tissue_ids']",
    )
    tissue_mapping = _ordered_tissue_mapping_from_arrays(
        prepared_mesh["ordered_tissue_ids"],
        prepared_mesh["ordered_tissues"],
    )
    ordered_tissue_ids, ordered_tissues = ordered_tissue_arrays(tissue_mapping)
    unknown_ids = np.setdiff1d(np.unique(tissue_ids), ordered_tissue_ids)
    if unknown_ids.size:
        raise ValueError(
            "prepared_mesh['element_tissue_ids'] contains IDs not represented by ordered_tissues: "
            f"{unknown_ids.tolist()}"
        )
    return {
        "nodes": nodes,
        "elements": elements,
        "element_tissue_ids": tissue_ids,
        "ordered_tissue_ids": ordered_tissue_ids,
        "ordered_tissues": ordered_tissues,
    }


def validate_tissue_property_coverage(
    element_tissue_ids: ArrayLike,
    number_of_media: int,
) -> None:
    """Require every element ID to reference an available MMC medium."""
    tissue_ids = np.asarray(element_tissue_ids)
    if tissue_ids.size == 0 or tissue_ids.min() < 0 or tissue_ids.max() >= number_of_media:
        raise ValueError("prepared_mesh['element_tissue_ids'] contains IDs not represented by the optical properties")


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


def validate_mesh_settings(experiment_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate mesh configuration and return canonical preparation values."""
    mesh_settings = require_config_section(experiment_config, "mesh_settings", _MESH_SETTINGS_KEYS)
    ordered_tissue_ids, ordered_tissues = ordered_tissue_arrays(mesh_settings["ordered_tissues"])

    units = mesh_settings["mesh_units"]
    try:
        normalized_units = units.lower()
    except AttributeError as error:
        raise ValueError("mesh_units must be either 'mm', 'cm' or 'm'") from error
    unit_scales = {"mm": 1.0, "cm": 10.0, "m": 1_000.0}
    try:
        unit_scale = unit_scales[normalized_units]
    except KeyError as error:
        raise ValueError("mesh_units must be either 'mm', 'cm' or 'm'") from error

    orientation = mesh_settings["mesh_orientation"]
    try:
        orientation_matrix = make_orientation_matrices()[orientation.upper()]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"Unknown mesh orientation {orientation!r}") from error

    return {
        "ordered_tissue_ids": ordered_tissue_ids,
        "ordered_tissues": ordered_tissues,
        "unit_scale": unit_scale,
        "orientation_matrix": orientation_matrix,
    }


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
