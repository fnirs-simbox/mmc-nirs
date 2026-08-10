import numpy as np

from mmc_nirs.registration import find_optode_directions, make_orientation_matrices


def test_orientation_matrices_contain_all_valid_orientations() -> None:
    matrices = make_orientation_matrices()

    assert len(matrices) == 48
    np.testing.assert_array_equal(matrices["RAS"], np.eye(3))
    np.testing.assert_array_equal(matrices["LPS"], np.diag([-1.0, -1.0, 1.0]))


def test_orientation_matrix_handles_permuted_axes() -> None:
    matrix = make_orientation_matrices()["ASR"]

    transformed = np.array([[2.0, 3.0, 5.0]]) @ matrix.T

    np.testing.assert_array_equal(transformed, [[5.0, 2.0, 3.0]])


def test_find_optode_directions_uses_mesh_location_not_only_extent() -> None:
    mesh_nodes = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    optodes = np.array([[20.0, 15.0, 15.0], [15.0, 10.0, 15.0]])

    directions = find_optode_directions(optodes, mesh_nodes)

    np.testing.assert_allclose(directions, [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
