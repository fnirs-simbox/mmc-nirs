import importlib

import numpy as np

from mmc_nirs.light_transport import find_optode_directions, make_orientation_matrices
from mmc_nirs.utils.mesh_utils import make_orientation_matrices as mesh_orientation_matrices

register_module = importlib.import_module("mmc_nirs.light_transport.register_probe")


def test_light_transport_reexports_orientation_matrices() -> None:
    assert make_orientation_matrices is mesh_orientation_matrices


def test_find_optode_directions_uses_mesh_location_not_only_extent() -> None:
    mesh_nodes = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    optodes = np.array([[20.0, 15.0, 15.0], [15.0, 10.0, 15.0]])

    directions = find_optode_directions(optodes, mesh_nodes)

    np.testing.assert_allclose(directions, [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_register_probe_converts_centimetres_to_millimetres(monkeypatch) -> None:
    monkeypatch.setattr(
        register_module.trimesh.registration,
        "icp",
        lambda points, mesh_nodes: (np.eye(4), points, 0.0),
    )
    monkeypatch.setattr(
        register_module,
        "_embed_optodes",
        lambda coordinates, *args: (coordinates, np.zeros(len(coordinates), dtype=np.intp)),
    )

    registered_sources, registered_detectors, *_ = register_module.register_probe(
        [[0, 0, 0]],
        [[1, 0, 0]],
        [[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, 20]],
        [[0, 1, 2, 3]],
        probe_units="cm",
    )

    np.testing.assert_allclose(registered_detectors - registered_sources, [[10, 0, 0]])
