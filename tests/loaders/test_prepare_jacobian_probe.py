import h5py
import numpy as np
import pytest

from mmc_nirs.loaders.prepare_jacobian_probe import load_channel_pairs_from_snirf, prepare_jacobian_probe


def test_prepare_jacobian_probe_returns_probe() -> None:
    sources = [[0, 0, 0], [10, 0, 0]]
    detectors = [[1, 0, 0], [30, 0, 0]]
    pairings = [[1, 1], [2, 2]]

    probe = prepare_jacobian_probe(sources, detectors, "cm", "RAS", pairings, "distance", 15.0)

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
def test_prepare_jacobian_probe_rejects_invalid_arrays(sources, detectors, flag, argument, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        prepare_jacobian_probe(sources, detectors, "mm", "RAS", [[0, 0]], flag, argument)


def test_load_channel_pairs_from_snirf_sorts_measurement_lists_numerically(tmp_path) -> None:
    snirf_path = tmp_path / "probe.snirf"
    with h5py.File(snirf_path, "w") as snirf:
        data_group = snirf.create_group("nirs").create_group("data1")
        for measurement_number, source_index, detector_index in ((10, 3, 4), (2, 1, 2)):
            measurement = data_group.create_group(f"measurementList{measurement_number}")
            measurement.create_dataset("sourceIndex", data=source_index)
            measurement.create_dataset("detectorIndex", data=detector_index)

    pairings = load_channel_pairs_from_snirf(snirf_path)

    np.testing.assert_array_equal(pairings, [[1, 2], [3, 4]])
