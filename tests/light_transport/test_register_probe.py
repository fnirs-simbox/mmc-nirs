import importlib

import numpy as np

from mmcnirs.light_transport import find_optode_directions, make_orientation_matrices
from mmcnirs.utils.mesh_utils import make_orientation_matrices as mesh_orientation_matrices

register_module = importlib.import_module("mmcnirs.light_transport.register_probe")


def test_light_transport_reexports_orientation_matrices() -> None:
    assert make_orientation_matrices is mesh_orientation_matrices


def test_find_optode_directions_uses_mesh_location_not_only_extent() -> None:
    mesh_nodes = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    optodes = np.array([[20.0, 15.0, 15.0], [15.0, 10.0, 15.0]])

    directions = find_optode_directions(optodes, mesh_nodes)

    np.testing.assert_allclose(directions, [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_register_probe_converts_centimetres_to_millimetres(monkeypatch) -> None:
    monkeypatch.setattr(
        register_module,
        "_minimize_surface_translation",
        lambda points, mesh_nodes, mesh_elements: points,
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


def test_register_probe_roughly_aligns_probe_and_mesh_tops(monkeypatch) -> None:
    points_passed_to_translation = None

    def capture_translation(points, mesh_nodes, mesh_elements):
        nonlocal points_passed_to_translation
        points_passed_to_translation = points
        return points

    monkeypatch.setattr(register_module, "_minimize_surface_translation", capture_translation)
    monkeypatch.setattr(
        register_module,
        "_embed_optodes",
        lambda coordinates, *args: (coordinates, np.zeros(len(coordinates), dtype=np.intp)),
    )

    register_module.register_probe(
        [[0, 0, 2]],
        [[2, 2, 6]],
        [[10, 20, 30], [20, 20, 30], [10, 40, 30], [10, 20, 50]],
        [[0, 1, 2, 3]],
    )

    assert points_passed_to_translation is not None
    np.testing.assert_allclose(points_passed_to_translation.min(axis=0)[:2], [14, 29])
    np.testing.assert_allclose(points_passed_to_translation[:, 2].max(), 50)


def test_surface_registration_applies_only_one_translation(monkeypatch) -> None:
    class SuccessfulResult:
        success = True
        message = ""
        x = np.array([3.0, -2.0, 1.0])

    monkeypatch.setattr(register_module, "minimize", lambda *args, **kwargs: SuccessfulResult())
    coordinates = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    registered = register_module._minimize_surface_translation(coordinates, nodes, np.array([[0, 1, 2, 3]]))

    np.testing.assert_allclose(registered - coordinates, [[3, -2, 1], [3, -2, 1]])
