import numpy as np
import pytest

from mmcnirs.utils.mesh_utils import (
    _find_containing_elements,
    as_coordinate_array,
    as_element_array,
    as_element_tissue_id_array,
    as_node_tissue_id_array,
    find_closest_nodes,
    make_orientation_matrices,
    ordered_tissue_arrays,
    repair_node_tissue_ids,
    validate_mesh_settings,
    validate_prepared_mesh,
    validate_tissue_property_coverage,
)


def test_validate_mesh_settings_returns_canonical_preparation_values() -> None:
    settings = validate_mesh_settings(
        {
            "mesh_settings": {
                "ordered_tissues": {"1": "scalp", "0": "ambient_air"},
                "mesh_orientation": "lia",
                "mesh_units": "cm",
            }
        }
    )

    np.testing.assert_array_equal(settings["ordered_tissue_ids"], [0, 1])
    np.testing.assert_array_equal(settings["ordered_tissues"], ["ambient_air", "scalp"])
    assert settings["unit_scale"] == 10.0
    np.testing.assert_array_equal(
        settings["orientation_matrix"],
        make_orientation_matrices()["LIA"],
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


def test_ordered_tissue_arrays_use_explicit_numeric_ids() -> None:
    tissue_ids, tissue_names = ordered_tissue_arrays({"2": "skull", "0": "ambient_air", "1": "scalp"})

    np.testing.assert_array_equal(tissue_ids, [0, 1, 2])
    np.testing.assert_array_equal(tissue_names, ["ambient_air", "scalp", "skull"])
    with pytest.raises(ValueError, match="duplicate tissue names"):
        ordered_tissue_arrays({"0": "ambient_air", "1": "scalp", "2": "scalp"})


@pytest.mark.parametrize(
    "ordered_tissues",
    [
        {"0": "ambient_air", "2": "scalp"},
        {"00": "ambient_air", "1": "scalp"},
        {"background": "ambient_air", "1": "scalp"},
    ],
)
def test_ordered_tissue_arrays_reject_invalid_ids(ordered_tissues) -> None:
    with pytest.raises(ValueError, match="ordered_tissues"):
        ordered_tissue_arrays(ordered_tissues)


def test_as_element_tissue_id_array_requires_one_integer_id_per_element() -> None:
    np.testing.assert_array_equal(as_element_tissue_id_array([1.0, 2.0], 2), [1, 2])
    with pytest.raises(ValueError, match="one value per element"):
        as_element_tissue_id_array([1], 2)
    with pytest.raises(ValueError, match="integer IDs"):
        as_element_tissue_id_array([1.5], 1)


def test_as_node_tissue_id_array_requires_one_integer_id_per_node() -> None:
    np.testing.assert_array_equal(as_node_tissue_id_array([1.0, 2.0], 2), [1, 2])
    with pytest.raises(ValueError, match="one value per node"):
        as_node_tissue_id_array([1], 2)
    with pytest.raises(ValueError, match="integer IDs"):
        as_node_tissue_id_array([1.5], 1)


def test_validate_prepared_mesh_returns_canonical_copies() -> None:
    prepared = validate_prepared_mesh(
        {
            "nodes": np.eye(3, 4).T,
            "elements": [[0, 1, 2, 3]],
            "node_tissue_ids": [0, 1, 1, 1],
            "element_tissue_ids": [1],
            "ordered_tissue_ids": [1, 0],
            "ordered_tissues": ["tissue", "ambient_air"],
        }
    )

    np.testing.assert_array_equal(prepared["elements"], [[0, 1, 2, 3]])
    np.testing.assert_array_equal(prepared["node_tissue_ids"], [0, 1, 1, 1])
    np.testing.assert_array_equal(prepared["element_tissue_ids"], [1])
    np.testing.assert_array_equal(prepared["ordered_tissue_ids"], [0, 1])
    np.testing.assert_array_equal(prepared["ordered_tissues"], ["ambient_air", "tissue"])


def test_validate_prepared_mesh_rejects_unrepresented_element_tissue_ids() -> None:
    with pytest.raises(ValueError, match="IDs not represented by ordered_tissues"):
        validate_prepared_mesh(
            {
                "nodes": np.eye(3, 4).T,
                "elements": [[0, 1, 2, 3]],
                "node_tissue_ids": [0, 1, 1, 1],
                "element_tissue_ids": [2],
                "ordered_tissue_ids": [0, 1],
                "ordered_tissues": ["ambient_air", "tissue"],
            }
        )


def test_validate_prepared_mesh_rejects_unrepresented_node_tissue_ids() -> None:
    with pytest.raises(ValueError, match="IDs not represented by ordered_tissues"):
        validate_prepared_mesh(
            {
                "nodes": np.eye(3, 4).T,
                "elements": [[0, 1, 2, 3]],
                "node_tissue_ids": [0, 1, 1, -1],
                "element_tissue_ids": [1],
                "ordered_tissue_ids": [0, 1],
                "ordered_tissues": ["ambient_air", "tissue"],
            }
        )


def test_repair_node_tissue_ids_uses_nearest_connected_valid_node_and_reports_summary() -> None:
    nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])

    with pytest.warns(UserWarning) as warnings:
        repaired = repair_node_tissue_ids(
            nodes,
            np.array([[0, 1, 2, 3]]),
            np.array([4, 2, -1, 3]),
            np.array([2, 3, 4]),
        )

    np.testing.assert_array_equal(repaired, [4, 2, 4, 3])
    summary = str(warnings[0].message)
    assert "node_id nearest_neighbor_id distance_mm old_label new_label" in summary
    assert "2 0 1.000000 -1 4" in summary


def test_repair_node_tissue_ids_does_not_cascade_repaired_labels() -> None:
    nodes = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0]])
    elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

    with pytest.warns(UserWarning) as warnings:
        repaired = repair_node_tissue_ids(nodes, elements, np.array([-1, -1, 2, 3, 3]), np.array([2, 3]))

    np.testing.assert_array_equal(repaired, [2, 2, 2, 3, 3])
    assert "1 2 9.900000 -1 2" in str(warnings[0].message)


def test_repair_node_tissue_ids_falls_back_to_global_node_and_warns_for_long_distance() -> None:
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [10.0, 0.0, 0.0]])

    with pytest.warns(UserWarning) as warnings:
        repaired = repair_node_tissue_ids(
            nodes,
            np.array([[0, 1, 2, 3]]),
            np.array([2, 2, 2, 2, -1]),
            np.array([2]),
        )

    np.testing.assert_array_equal(repaired, [2, 2, 2, 2, 2])
    assert len(warnings) == 2
    assert "exceeds the mesh mean edge distance" in str(warnings[1].message)


def test_repair_node_tissue_ids_rejects_mesh_without_a_valid_node_label() -> None:
    with pytest.raises(ValueError, match="no IDs represented"):
        repair_node_tissue_ids(
            np.eye(3, 4).T,
            np.array([[0, 1, 2, 3]]),
            np.full(4, -1),
            np.array([0, 1]),
        )


def test_validate_tissue_property_coverage_rejects_missing_media() -> None:
    validate_tissue_property_coverage([0, 1, 2], 3)
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
