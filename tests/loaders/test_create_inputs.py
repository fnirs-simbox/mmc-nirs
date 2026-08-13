import numpy as np
import pytest

from mmc_nirs.loaders.create_inputs import create_mesh_file, create_probe_input


def test_create_probe_input_returns_probe_without_saving() -> None:
    sources = [[0, 0, 0], [10, 0, 0]]
    detectors = [[1, 0, 0], [30, 0, 0]]
    pairings = [[1, 1], [2, 2]]

    probe = create_probe_input(sources, detectors, "cm", "RAS", pairings, "distance", 15.0)

    assert set(probe) == {
        "source_positions",
        "detector_positions",
        "orientation",
        "channel_pairings",
        "short_separation_indices",
    }
    np.testing.assert_array_equal(probe["source_positions"], np.asarray(sources) * 10)
    np.testing.assert_array_equal(probe["detector_positions"], np.asarray(detectors) * 10)
    np.testing.assert_array_equal(probe["short_separation_indices"], [0])
    assert probe["orientation"] == "RAS"


def test_create_probe_input_saves_loadable_npz(tmp_path) -> None:
    output = tmp_path / "probe.npz"

    returned = create_probe_input([[1, 2, 3]], [[4, 5, 6]], "mm", "RAS", [[0, 0]], "index", [0], True, output)

    assert returned is None
    with np.load(output) as saved:
        assert set(saved.files) == {
            "source_positions",
            "detector_positions",
            "orientation",
            "channel_pairings",
            "short_separation_indices",
        }
        np.testing.assert_array_equal(saved["short_separation_indices"], [0])


@pytest.mark.parametrize(
    ("sources", "detectors", "flag", "argument", "message"),
    [
        ([], [[1, 2, 3]], "index", [0], "source_positions"),
        ([[1, 2, 3]], [[1, 2]], "index", [0], "detector_positions"),
        ([[1, 2, 3]], [[1, 2, 3]], "invalid", [0], "short_separation_flag"),
        ([[1, 2, 3]], [[1, 2, 3]], "distance", 1, "finite float"),
        ([[1, 2, 3]], [[1, 2, 3]], "index", (0,), "list of integers"),
    ],
)
def test_create_probe_input_rejects_invalid_arrays(sources, detectors, flag, argument, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        create_probe_input(sources, detectors, "mm", "RAS", [[0, 0]], flag, argument)


def test_create_probe_input_rejects_non_npz_filename(tmp_path) -> None:
    with pytest.raises(ValueError, match=".npz"):
        create_probe_input([[1, 2, 3]], [[4, 5, 6]], "mm", "RAS", [[0, 0]], "index", [0], True, tmp_path / "probe.mat")


def test_create_mesh_file_normalizes_and_saves_mesh(tmp_path) -> None:
    output = tmp_path / "mesh.npz"
    nodes = np.array(
        [
            [0.001, 0.002, 0.003],
            [0.004, 0.005, 0.006],
            [0.007, 0.008, 0.009],
            [0.010, 0.011, 0.012],
        ]
    )

    create_mesh_file(
        output,
        nodes,
        [[1, 2, 3, 4]],
        [50, 40, 30, 20],
        orientation="LIA",
        units="m",
        scalp_idx=10,
        skull_idx=20,
        CSF_idx=30,
        gray_matter_idx=40,
        white_matter_idx=50,
        save_mesh=True,
    )

    with np.load(output) as mesh:
        np.testing.assert_allclose(mesh["nodes"], nodes * 1000 @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]).T)
        np.testing.assert_array_equal(mesh["elements"], [[0, 1, 2, 3]])
        np.testing.assert_array_equal(mesh["node_tissue_values"], [1, 2, 3, 4])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("nodes", [[0, 0]], "nodes"),
        ("elements", [[0, 1, 2]], "elements"),
        ("node_tissue_values", [1, 2, 3], "node_tissue_values"),
        ("orientation", "XYZ", "orientation"),
        ("units", "km", "units"),
    ],
)
def test_create_mesh_file_rejects_invalid_input(tmp_path, field, value, message) -> None:
    arguments = {
        "filename": tmp_path / "mesh.npz",
        "nodes": np.zeros((4, 3)),
        "elements": [[0, 1, 2, 3]],
        "node_tissue_values": [1, 2, 3, 4],
        "orientation": "RAS",
        "units": "mm",
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        create_mesh_file(**arguments)


def test_create_mesh_file_rejects_duplicate_or_unknown_tissue_labels(tmp_path) -> None:
    arguments = (
        tmp_path / "mesh.npz",
        np.zeros((4, 3)),
        [[0, 1, 2, 3]],
        [1, 2, 3, 99],
        "RAS",
        "mm",
    )
    with pytest.raises(ValueError, match="unknown tissue labels"):
        create_mesh_file(*arguments)

    with pytest.raises(ValueError, match="must be unique"):
        create_mesh_file(*arguments, scalp_idx=1)
