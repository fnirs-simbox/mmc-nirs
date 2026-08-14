import numpy as np

from mmc_nirs.registration import find_optode_directions, make_orientation_matrices
from mmc_nirs.utils.mesh_utils import make_orientation_matrices as mesh_orientation_matrices


def test_registration_reexports_orientation_matrices() -> None:
    assert make_orientation_matrices is mesh_orientation_matrices


def test_find_optode_directions_uses_mesh_location_not_only_extent() -> None:
    mesh_nodes = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    optodes = np.array([[20.0, 15.0, 15.0], [15.0, 10.0, 15.0]])

    directions = find_optode_directions(optodes, mesh_nodes)

    np.testing.assert_allclose(directions, [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
