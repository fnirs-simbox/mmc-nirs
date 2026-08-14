import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from mmc_nirs import load_config, load_light_transport_results


@pytest.fixture
def experiment_config_path(tmp_path: Path) -> Path:
    experiment_directory = tmp_path / "pain"
    experiment_directory.mkdir()
    np.savez(experiment_directory / "mesh.npz", nodes=np.zeros((50, 3)))
    np.savez(
        experiment_directory / "probe.npz",
        sourcepos=np.zeros((8, 3)),
        detpos=np.zeros((16, 3)),
        detnorms=np.zeros((16, 3)),
        short_separation_indices=np.array([0, 2]),
        long_separation_indices=np.array([1, 3]),
    )
    for wavelength in (690, 830):
        np.savez(
            experiment_directory / f"jacobian_{wavelength}.npz",
            J=np.zeros((100, 50)),
            mea0=np.zeros(100),
            channelidx=np.arange(1, 13),
        )
    np.save(experiment_directory / "activation_map.npy", np.zeros(50))
    config_path = experiment_directory / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "Pain",
                "filepaths": {
                    "meshfile": "mesh.npz",
                    "nodes_var": "nodes",
                    "jacobians": ["jacobian_690.npz", "jacobian_830.npz"],
                    "probefile": "probe.npz",
                    "activation_map": "activation_map.npy",
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_light_transport_results_loads_simulator_inputs(experiment_config_path: Path) -> None:
    data = load_light_transport_results(load_config(experiment_config_path))

    assert data["nodes"].shape == (50, 3)
    assert data["source_positions"].shape == (8, 3)
    assert data["detector_positions"].shape == (16, 3)
    assert data["detector_norms"].shape == (16, 3)
    np.testing.assert_array_equal(data["short_separation_indices"], [0, 2])
    np.testing.assert_array_equal(data["long_separation_indices"], [1, 3])
    assert len(data["jacobian_list"]) == 2
    assert all(jacobian.shape == (12, 50) for jacobian in data["jacobian_list"])
    assert all(measurements.shape == (12, 1) for measurements in data["measurements_zero_list"])
    assert data["channel_idx"].shape == (12,)
    assert np.all(data["channel_idx"] >= 0)
    assert np.all(data["channel_idx"] < 100)
    assert np.all(np.diff(data["channel_idx"]) > 0)


def test_load_light_transport_results_can_skip_jacobians(experiment_config_path: Path) -> None:
    data = load_light_transport_results(load_config(experiment_config_path), use_jacobian=False)

    assert data["jacobian_list"] == []
    assert data["measurements_zero_list"] == []
    assert data["channel_idx"] is None
    assert data["activation_map"] is None


def test_load_light_transport_results_loads_external_experiment(tmp_path, experiment_config_path: Path) -> None:
    experiment_directory = tmp_path / "finger_tapping"
    shutil.copytree(experiment_config_path.parent, experiment_directory)

    config_path = experiment_directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["name"] = "Finger Tapping"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    data = load_light_transport_results(load_config(config_path))

    assert data["nodes"].shape == (50, 3)
    assert len(data["jacobian_list"]) == 2
