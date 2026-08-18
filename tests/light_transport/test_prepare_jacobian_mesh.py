import numpy as np
import pytest

from mmc_nirs.light_transport.prepare_jacobian_mesh import prepare_jacobian_mesh


@pytest.fixture
def experiment_config(tmp_path):
    return {
        "experiment_dir": tmp_path / "prepared",
        "filepaths": {
            "meshfile": "mesh.npz",
        },
    }


def test_prepare_jacobian_mesh_returns_normalized_mesh(experiment_config) -> None:
    prepared = prepare_jacobian_mesh(
        np.zeros((4, 3)),
        [[0, 1, 2, 3]],
        [1],
        orientation="RAS",
        units="mm",
        experiment_config=experiment_config,
    )

    assert set(prepared) == {"nodes", "elements", "element_tissue_values"}
    np.testing.assert_array_equal(prepared["elements"], [[0, 1, 2, 3]])
    np.testing.assert_array_equal(prepared["element_tissue_values"], [1])


def test_prepare_jacobian_mesh_reorients_and_remaps_tissues(experiment_config) -> None:
    nodes = np.array(
        [
            [0.001, 0.002, 0.003],
            [0.004, 0.005, 0.006],
            [0.007, 0.008, 0.009],
            [0.010, 0.011, 0.012],
        ]
    )

    mesh = prepare_jacobian_mesh(
        nodes,
        np.tile([[1, 2, 3, 4]], (4, 1)),
        [50, 40, 30, 20],
        orientation="LIA",
        units="m",
        experiment_config=experiment_config,
        scalp_idx=10,
        skull_idx=20,
        CSF_idx=30,
        gray_matter_idx=40,
        white_matter_idx=50,
    )

    np.testing.assert_allclose(mesh["nodes"], nodes * 1000 @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]).T)
    np.testing.assert_array_equal(mesh["elements"], np.tile([[0, 1, 2, 3]], (4, 1)))
    np.testing.assert_array_equal(mesh["element_tissue_values"], [1, 2, 3, 4])


def test_prepare_jacobian_mesh_saves_to_configured_path(experiment_config) -> None:
    prepared = prepare_jacobian_mesh(
        np.zeros((4, 3)),
        [[0, 1, 2, 3]],
        [1],
        "RAS",
        "mm",
        experiment_config,
        save_mesh=True,
    )

    output_path = experiment_config["experiment_dir"] / "mesh.npz"
    assert output_path.is_file()
    with np.load(output_path, allow_pickle=False) as archive:
        for key, value in prepared.items():
            np.testing.assert_array_equal(archive[key], value)


def test_prepare_jacobian_mesh_reuses_existing_archive_before_validation(experiment_config) -> None:
    output_dir = experiment_config["experiment_dir"]
    output_dir.mkdir()
    cached = {
        "nodes": np.ones((4, 3)),
        "elements": np.array([[0, 1, 2, 3]]),
        "element_tissue_values": np.array([1]),
    }
    np.savez(output_dir / "mesh.npz", **cached)

    prepared = prepare_jacobian_mesh([], [], [], "invalid", "invalid", experiment_config)

    for key, value in cached.items():
        np.testing.assert_array_equal(prepared[key], value)


def test_prepare_jacobian_mesh_overwrites_existing_archive(experiment_config) -> None:
    output_dir = experiment_config["experiment_dir"]
    output_dir.mkdir()
    np.savez(
        output_dir / "mesh.npz",
        nodes=np.ones((4, 3)),
        elements=[[0, 1, 2, 3]],
        element_tissue_values=[1],
    )

    prepared = prepare_jacobian_mesh(
        np.zeros((4, 3)),
        [[0, 1, 2, 3]],
        [1],
        "RAS",
        "mm",
        experiment_config,
        save_mesh=True,
        overwrite=True,
    )

    np.testing.assert_array_equal(prepared["nodes"], np.zeros((4, 3)))
    with np.load(output_dir / "mesh.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["nodes"], np.zeros((4, 3)))


def test_prepare_jacobian_mesh_rejects_incompatible_cache(experiment_config) -> None:
    output_dir = experiment_config["experiment_dir"]
    output_dir.mkdir()
    np.savez(output_dir / "mesh.npz", nodes=np.zeros((4, 3)))

    with pytest.raises(ValueError, match="missing required field"):
        prepare_jacobian_mesh([], [], [], "RAS", "mm", experiment_config)


def test_prepare_jacobian_mesh_rejects_non_npz_output_name(experiment_config) -> None:
    experiment_config["filepaths"]["meshfile"] = "mesh.mat"

    with pytest.raises(ValueError, match=".npz"):
        prepare_jacobian_mesh([], [], [], "RAS", "mm", experiment_config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("nodes", [[0, 0]], "nodes"),
        ("elements", [[0, 1, 2]], "elements"),
        ("element_tissue_values", [1, 2], "element_tissue_values"),
        ("orientation", "XYZ", "orientation"),
        ("units", "km", "units"),
    ],
)
def test_prepare_jacobian_mesh_rejects_invalid_input(experiment_config, field, value, message) -> None:
    arguments = {
        "nodes": np.zeros((4, 3)),
        "elements": [[0, 1, 2, 3]],
        "element_tissue_values": [1],
        "orientation": "RAS",
        "units": "mm",
        "experiment_config": experiment_config,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        prepare_jacobian_mesh(**arguments)


def test_prepare_jacobian_mesh_rejects_duplicate_or_unknown_tissue_labels(experiment_config) -> None:
    arguments = (
        np.zeros((4, 3)),
        [[0, 1, 2, 3]],
        [99],
        "RAS",
        "mm",
        experiment_config,
    )
    with pytest.raises(ValueError, match="unknown tissue labels"):
        prepare_jacobian_mesh(*arguments)

    with pytest.raises(ValueError, match="must be unique"):
        prepare_jacobian_mesh(*arguments, scalp_idx=1)
