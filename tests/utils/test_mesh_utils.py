import numpy as np
import pytest

from mmc_nirs.utils.mesh_utils import (
    _find_containing_elements,
    as_coordinate_array,
    as_element_array,
    as_element_tissue_array,
    find_closest_nodes,
    make_orientation_matrices,
    validate_prepared_mesh,
    validate_tissue_property_coverage,
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
    coordinates = as_coordinate_array([[1, 2, 3]], "coordinates")

    np.testing.assert_array_equal(coordinates, [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="shape"):
        as_coordinate_array([[1, 2]], "coordinates")
    with pytest.raises(ValueError, match="finite"):
        as_coordinate_array([[1, 2, np.inf]], "coordinates")


def test_as_element_array_normalizes_one_based_indices() -> None:
    elements = as_element_array([[1, 2, 3, 4, 9]], 4)

    np.testing.assert_array_equal(elements, [[0, 1, 2, 3]])


def test_as_element_tissue_array_requires_one_integer_label_per_element() -> None:
    np.testing.assert_array_equal(as_element_tissue_array([1.0, 2.0], 2), [1, 2])
    with pytest.raises(ValueError, match="one value per element"):
        as_element_tissue_array([1], 2)
    with pytest.raises(ValueError, match="integer labels"):
        as_element_tissue_array([1.5], 1)


def test_validate_prepared_mesh_returns_canonical_copies() -> None:
    prepared = validate_prepared_mesh(
        {
            "nodes": np.eye(3, 4).T,
            "elements": [[0, 1, 2, 3]],
            "element_tissue_values": [1],
        }
    )

    np.testing.assert_array_equal(prepared["elements"], [[0, 1, 2, 3]])
    np.testing.assert_array_equal(prepared["element_tissue_values"], [1])


def test_validate_tissue_property_coverage_rejects_missing_media() -> None:
    validate_tissue_property_coverage([1, 2], 3)
    with pytest.raises(ValueError, match="not represented"):
        validate_tissue_property_coverage([1, 3], 3)


def test_find_containing_elements_handles_interior_and_exterior_points() -> None:
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

    containing = _find_containing_elements(
        np.array([[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]]),
        nodes,
        np.array([[0, 1, 2, 3]]),
    )

    np.testing.assert_array_equal(containing, [0, -1])


def test_find_closest_nodes_returns_one_index_per_target() -> None:
    indices = find_closest_nodes(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]],
    )

    np.testing.assert_array_equal(indices, [0, 1])
