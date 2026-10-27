import numpy as np
import pytest
import trimesh

from mmcnirs.utils.mesh_utils import make_surface_mesh
from mmcnirs.utils.probe_utils import find_smoothed_surface_directions, inward_surface_face_normals


def test_radius_uses_connected_area_weighted_patch_not_nearby_disconnected_sheet():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 0, 1],
            [2, 1, 1],
            [0.1, 0.1, 0.05],
            [0.1, 0.8, 0.05],
            [0.1, 0.1, 0.8],
            [0, 0, -10],
        ],
        dtype=float,
    )
    surface = trimesh.Trimesh(
        vertices=vertices, faces=[[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3], [6, 7, 8]], process=False
    )
    points = np.array([[0.2, 0.2, 0]])
    original = points.copy()
    small, _ = find_smoothed_surface_directions(surface, points, vertices, surface_radius_mm=0.1)
    large, spread = find_smoothed_surface_directions(surface, points, vertices, surface_radius_mm=3)
    np.testing.assert_allclose(small, [[0, 0, -1]])
    np.testing.assert_allclose(large, np.array([[1, 0, -2]]) / np.sqrt(5))
    np.testing.assert_allclose(np.linalg.norm(large, axis=1), 1)
    np.testing.assert_array_equal(points, original)
    assert 0 < spread[0] < 45


def test_owning_tetrahedron_sign_is_independent_of_winding_and_head_center():
    # The far component shifts the global center outside the first tetrahedron.
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [100, 0, 0], [101, 0, 0], [100, 1, 0], [100, 0, 1]], dtype=float
    )
    elements = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    surface = make_surface_mesh(nodes, elements)
    points = [[1 / 3, 1 / 3, 1 / 3]]
    expected = np.array([[-1, -1, -1]]) / np.sqrt(3)
    for flip in (False, True):
        candidate = trimesh.Trimesh(
            vertices=nodes, faces=surface.faces[:, ::-1] if flip else surface.faces, process=False
        )
        directions, _ = find_smoothed_surface_directions(candidate, points, nodes, 0.01, mesh_elements=elements)
        np.testing.assert_allclose(directions, expected)
        normals, areas = inward_surface_face_normals(candidate, nodes, elements)
        assert np.all(areas > 0)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)


@pytest.mark.parametrize("radius", [0, -1, np.inf, np.nan, True, "15", np.array([15.0])])
def test_surface_radius_must_be_a_positive_finite_scalar(radius):
    nodes = np.array([[0, 0, 0], [30, 0, 0], [0, 30, 0], [0, 0, 30]], dtype=float)
    surface = make_surface_mesh(nodes, np.array([[0, 1, 2, 3]]))
    with pytest.raises(ValueError, match="surface_radius_mm"):
        find_smoothed_surface_directions(surface, [[1, 1, 0]], nodes, radius)


def test_default_radius_is_explicit_fifteen_mm():
    nodes = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=float)
    elements = np.array([[0, 1, 2, 3]])
    surface = make_surface_mesh(nodes, elements)
    default, _ = find_smoothed_surface_directions(surface, [[20, 20, 0]], nodes, mesh_elements=elements)
    explicit, _ = find_smoothed_surface_directions(surface, [[20, 20, 0]], nodes, 15, mesh_elements=elements)
    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_allclose(default, [[0, 0, 1]])
