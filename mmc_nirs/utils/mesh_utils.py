"""Shared validation and geometry helpers for tetrahedral meshes."""

import itertools

import numpy as np
from numpy.typing import ArrayLike


def _as_coordinate_array(values: ArrayLike, name: str) -> np.ndarray:
    """Return a finite, copied ``(n, 3)`` floating-point coordinate array."""
    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] == 0 or coordinates.shape[1] != 3:
        raise ValueError(f"{name} must be a non-empty array with shape (n, 3)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} must contain only finite values")
    return coordinates.copy()


def _as_element_array(
    values: ArrayLike,
    number_of_nodes: int,
    name: str = "mesh_elements",
    *,
    allow_extra_columns: bool = True,
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

    if elements.min() >= 1 and elements.max() <= number_of_nodes:
        elements -= 1
    if elements.min() < 0 or elements.max() >= number_of_nodes:
        raise ValueError(f"{name} contains an out-of-range vertex index")
    return elements


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


def find_closest_node(nodes: ArrayLike, target_point: ArrayLike) -> tuple[int, np.ndarray]:
    """Return the index and coordinates of the mesh node nearest a point."""
    node_array = np.asarray(nodes)
    target = np.asarray(target_point)
    if node_array.ndim != 2 or node_array.shape[0] == 0:
        raise ValueError("nodes must be a non-empty two-dimensional array")
    if target.shape != (node_array.shape[1],):
        raise ValueError("target_point must have one coordinate per node dimension")

    squared_distances = np.sum((node_array - target) ** 2, axis=1)
    closest_index = int(np.argmin(squared_distances))
    return closest_index, node_array[closest_index]
