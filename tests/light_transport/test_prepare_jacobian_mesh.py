import numpy as np
import pytest

from mmc_nirs.light_transport.prepare_jacobian_mesh import prepare_jacobian_mesh


@pytest.fixture
def experiment_config(tmp_path):
    return {
        "experiment_dir": tmp_path / "prepared",
        "ordered_tissues": {
            "0": "ambient_air",
            "1": "white_matter",
            "2": "gray_matter",
            "3": "CSF",
            "4": "skull",
            "5": "scalp",
        },
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

    assert set(prepared) == {
        "nodes",
        "elements",
        "element_tissue_ids",
        "ordered_tissue_ids",
        "ordered_tissues",
    }
    np.testing.assert_array_equal(prepared["elements"], [[0, 1, 2, 3]])
    np.testing.assert_array_equal(prepared["element_tissue_ids"], [1])
    np.testing.assert_array_equal(prepared["ordered_tissue_ids"], [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(prepared["ordered_tissues"], list(experiment_config["ordered_tissues"].values()))


def test_prepare_jacobian_mesh_reorients_and_preserves_ordered_tissue_ids(experiment_config) -> None:
    nodes = np.array(
        [
            [0.001, 0.002, 0.003],
            [0.004, 0.005, 0.006],
            [0.007, 0.008, 0.009],
            [0.010, 0.011, 0.012],
        ]
    )

    experiment_config["ordered_tissues"] = {
        "5": "white_matter",
        "2": "skull",
        "0": "ambient_air",
        "4": "gray_matter",
        "1": "scalp",
        "3": "CSF",
    }
    mesh = prepare_jacobian_mesh(
        nodes,
        np.tile([[1, 2, 3, 4]], (4, 1)),
        [5, 4, 3, 2],
        orientation="LIA",
        units="m",
        experiment_config=experiment_config,
    )

    np.testing.assert_allclose(mesh["nodes"], nodes * 1000 @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]).T)
    np.testing.assert_array_equal(mesh["elements"], np.tile([[0, 1, 2, 3]], (4, 1)))
    np.testing.assert_array_equal(mesh["element_tissue_ids"], [5, 4, 3, 2])
    np.testing.assert_array_equal(mesh["ordered_tissue_ids"], [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(
        mesh["ordered_tissues"],
        ["ambient_air", "scalp", "skull", "CSF", "gray_matter", "white_matter"],
    )


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
        "element_tissue_ids": np.array([1]),
        "ordered_tissue_ids": np.arange(6),
        "ordered_tissues": np.array(list(experiment_config["ordered_tissues"].values())),
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
        element_tissue_ids=[1],
        ordered_tissue_ids=np.arange(6),
        ordered_tissues=list(experiment_config["ordered_tissues"].values()),
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
        ("element_tissue_ids", [1, 2], "element_tissue_ids"),
        ("orientation", "XYZ", "orientation"),
        ("units", "km", "units"),
    ],
)
def test_prepare_jacobian_mesh_rejects_invalid_input(experiment_config, field, value, message) -> None:
    arguments = {
        "nodes": np.zeros((4, 3)),
        "elements": [[0, 1, 2, 3]],
        "element_tissue_ids": [1],
        "orientation": "RAS",
        "units": "mm",
        "experiment_config": experiment_config,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        prepare_jacobian_mesh(**arguments)


@pytest.mark.parametrize("invalid_id", [6, 99])
def test_prepare_jacobian_mesh_rejects_invalid_element_tissue_ids(experiment_config, invalid_id) -> None:
    arguments = (
        np.zeros((4, 3)),
        [[0, 1, 2, 3]],
        [invalid_id],
        "RAS",
        "mm",
        experiment_config,
    )
    with pytest.raises(ValueError, match="not represented by ordered_tissues"):
        prepare_jacobian_mesh(*arguments)


def test_prepare_jacobian_mesh_accepts_declared_background_id(experiment_config) -> None:
    prepared = prepare_jacobian_mesh(
        np.zeros((4, 3)),
        [[0, 1, 2, 3]],
        [0],
        "RAS",
        "mm",
        experiment_config,
    )

    np.testing.assert_array_equal(prepared["element_tissue_ids"], [0])


def test_prepare_jacobian_mesh_rejects_invalid_ordered_tissues(experiment_config) -> None:
    arguments = (
        np.zeros((4, 3)),
        [[0, 1, 2, 3]],
        [1],
        "RAS",
        "mm",
        experiment_config,
    )
    experiment_config["ordered_tissues"] = {"0": "ambient_air", "1": "scalp", "2": "scalp"}
    with pytest.raises(ValueError, match="duplicate tissue names"):
        prepare_jacobian_mesh(*arguments)

    experiment_config["ordered_tissues"] = {"0": "ambient_air", "2": "scalp"}
    with pytest.raises(ValueError, match="contiguous"):
        prepare_jacobian_mesh(*arguments)

    del experiment_config["ordered_tissues"]
    with pytest.raises(ValueError, match="missing required field: ordered_tissues"):
        prepare_jacobian_mesh(*arguments)
