import numpy as np
import pytest

from mmc_nirs.utils.mesh_utils import (
    _as_coordinate_array,
    _as_element_array,
    _find_containing_elements,
    find_closest_node,
    make_orientation_matrices,
)


def test_orientation_matrices_contain_all_valid_orientations() -> None:
    matrices = make_orientation_matrices()

    assert len(matrices) == 48
    np.testing.assert_array_equal(matrices["RAS"], np.eye(3))
    np.testing.assert_array_equal(matrices["LPS"], np.diag([-1.0, -1.0, 1.0]))


def test_orientation_matrix_handles_permuted_axes() -> None:
    matrix = make_orientation_matrices()["ASR"]

    transformed = np.array([[2.0, 3.0, 5.0]]) @ matrix.T

    np.testing.assert_array_equal(transformed, [[5.0, 2.0, 3.0]])


def test_as_coordinate_array_validates_shape_and_finite_values() -> None:
    coordinates = _as_coordinate_array([[1, 2, 3]], "coordinates")

    np.testing.assert_array_equal(coordinates, [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="shape"):
        _as_coordinate_array([[1, 2]], "coordinates")
    with pytest.raises(ValueError, match="finite"):
        _as_coordinate_array([[1, 2, np.inf]], "coordinates")


def test_as_element_array_normalizes_one_based_indices() -> None:
    elements = _as_element_array([[1, 2, 3, 4, 9]], 4)

    np.testing.assert_array_equal(elements, [[0, 1, 2, 3]])


def test_find_containing_elements_handles_interior_and_exterior_points() -> None:
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

    containing = _find_containing_elements(
        np.array([[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]]),
        nodes,
        np.array([[0, 1, 2, 3]]),
    )

    np.testing.assert_array_equal(containing, [0, -1])


def test_find_closest_node_returns_index_and_coordinates() -> None:
    nodes = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [5.0, 5.0, 5.0]])

    index, node = find_closest_node(nodes, [1.8, 1.1, 0.0])

    assert index == 1
    np.testing.assert_array_equal(node, nodes[1])


def test_find_closest_node_validates_target_shape() -> None:
    with pytest.raises(ValueError, match="one coordinate"):
        find_closest_node(np.zeros((2, 3)), [0.0, 0.0])
