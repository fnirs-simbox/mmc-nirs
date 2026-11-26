import importlib

import numpy as np
import pytest

from mmcnirs.utils.mesh_utils import make_surface_mesh
from mmcnirs.utils.probe_utils import find_center_directions

register_module = importlib.import_module("mmcnirs.light_transport.register_probe")


@pytest.fixture
def capture_registration(monkeypatch):
    calls = {}

    def translate(points, surface):
        calls["translated"] = points.copy()
        return points

    def project(points, surface, nodes, elements, epsilon):
        return points, points, np.zeros_like(points), np.zeros(len(points), dtype=int), np.zeros(len(points))

    def directions(surface, points, nodes, surface_radius_mm, **kwargs):
        calls["radius"] = surface_radius_mm
        calls["elements"] = kwargs["mesh_elements"]
        return np.tile([0.0, 0.0, -1.0], (len(points), 1)), np.zeros(len(points))

    monkeypatch.setattr(register_module, "_minimize_surface_translation", translate)
    monkeypatch.setattr(register_module, "project_optodes_just_inside", project)
    monkeypatch.setattr(register_module, "find_smoothed_surface_directions", directions)
    return calls


def test_find_optode_directions_uses_mesh_location_not_only_extent() -> None:
    mesh_nodes = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    optodes = np.array([[20.0, 15.0, 15.0], [15.0, 10.0, 15.0]])

    directions = find_center_directions(optodes, mesh_nodes)

    np.testing.assert_allclose(directions, [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_register_probe_converts_centimetres_to_millimetres(capture_registration) -> None:
    registered_sources, registered_detectors, *_ = register_module.register_probe(
        [[0, 0, 0]],
        [[1, 0, 0]],
        [[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, 20]],
        [[0, 1, 2, 3]],
        probe_units="cm",
    )

    np.testing.assert_allclose(registered_detectors - registered_sources, [[10, 0, 0]])
    assert capture_registration["radius"] == 15.0


def test_register_probe_roughly_aligns_probe_and_mesh_tops(capture_registration) -> None:
    register_module.register_probe(
        [[0, 0, 2]],
        [[2, 2, 6]],
        [[10, 20, 30], [20, 20, 30], [10, 40, 30], [10, 20, 50]],
        [[0, 1, 2, 3]],
    )

    points_passed_to_translation = capture_registration["translated"]
    np.testing.assert_allclose(points_passed_to_translation.min(axis=0)[:2], [14, 29])
    np.testing.assert_allclose(points_passed_to_translation[:, 2].max(), 50)


def test_register_probe_radius_only_changes_direction_request(capture_registration) -> None:
    args = ([[0, 0, 2]], [[2, 2, 6]], [[10, 20, 30], [20, 20, 30], [10, 40, 30], [10, 20, 50]], [[0, 1, 2, 3]])
    default = register_module.register_probe(*args)
    custom = register_module.register_probe(*args, surface_radius_mm=8.0)
    assert capture_registration["radius"] == 8.0
    np.testing.assert_array_equal(capture_registration["elements"], [[0, 1, 2, 3]])
    for index in (0, 1, 4, 5):
        np.testing.assert_array_equal(default[index], custom[index])


def test_surface_registration_applies_only_one_translation(monkeypatch) -> None:
    class SuccessfulResult:
        success = True
        message = ""
        x = np.array([3.0, -2.0, 1.0])

    monkeypatch.setattr(register_module, "minimize", lambda *args, **kwargs: SuccessfulResult())
    coordinates = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    registered = register_module._minimize_surface_translation(
        coordinates, make_surface_mesh(nodes, np.array([[0, 1, 2, 3]]))
    )

    np.testing.assert_allclose(registered - coordinates, [[3, -2, 1], [3, -2, 1]])
