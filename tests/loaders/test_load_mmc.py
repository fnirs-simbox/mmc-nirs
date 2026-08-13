import json
import shutil
from pathlib import Path

import numpy as np

from mmc_nirs import load_config, load_default_config, load_mmc_files


def test_load_mmc_files_loads_forward_model() -> None:
    data = load_mmc_files(load_default_config("pain"))

    assert data["nodes"].shape == (50, 3)
    assert data["source_positions"].shape == (8, 3)
    assert data["detector_positions"].shape == (16, 3)
    assert data["detector_norms"].shape == (16, 3)
    assert len(data["jacobian_list"]) == 2
    assert all(jacobian.shape == (12, 50) for jacobian in data["jacobian_list"])
    assert all(measurements.shape == (12, 1) for measurements in data["measurements_zero_list"])
    assert data["channel_idx"].shape == (12,)
    assert np.all(data["channel_idx"] >= 0)
    assert np.all(data["channel_idx"] < 100)
    assert np.all(np.diff(data["channel_idx"]) > 0)


def test_load_mmc_files_can_skip_jacobians() -> None:
    data = load_mmc_files(load_default_config("pain"), use_jacobian=False)

    assert data["jacobian_list"] == []
    assert data["measurements_zero_list"] == []
    assert data["channel_idx"] is None
    assert data["activation_map"] is None


def test_load_mmc_files_loads_external_experiment(tmp_path) -> None:
    bundled_directory = Path(__file__).parents[2] / "mmc_nirs" / "experiments" / "pain"
    experiment_directory = tmp_path / "finger_tapping"
    shutil.copytree(bundled_directory, experiment_directory)

    config_path = experiment_directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["name"] = "Finger Tapping"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    data = load_mmc_files(load_config(config_path))

    assert data["nodes"].shape == (50, 3)
    assert len(data["jacobian_list"]) == 2
