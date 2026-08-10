import numpy as np

from mmc_nirs import load_mmc_files


def test_load_mmc_files_loads_forward_model() -> None:
    data = load_mmc_files("pain")

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
    data = load_mmc_files("pain", use_jacobian=False)

    assert data["jacobian_list"] == []
    assert data["measurements_zero_list"] == []
    assert data["channel_idx"] is None
    assert data["activation_map"] is None
