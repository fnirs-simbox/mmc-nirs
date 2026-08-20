import h5py
import numpy as np
import pytest

from mmc_nirs.utils.probe_utils import (
    as_channel_pairing_array,
    as_unit_direction_array,
    flatten_channel_pairings,
    load_channel_pairs_from_snirf,
    normalize_channel_pairings,
    validate_probe_settings,
    validate_prepared_probe,
)


def test_load_channel_pairs_from_snirf_returns_unique_pairs_in_measurement_order(tmp_path) -> None:
    snirf_path = tmp_path / "probe.snirf"
    with h5py.File(snirf_path, "w") as snirf:
        data_group = snirf.create_group("nirs").create_group("data1")
        for measurement_number, source_index, detector_index in (
            (10, 3, 4),
            (3, 1, 2),
            (2, 1, 2),
        ):
            measurement = data_group.create_group(f"measurementList{measurement_number}")
            measurement.create_dataset("sourceIndex", data=[source_index])
            measurement.create_dataset("detectorIndex", data=[detector_index])

    pairings = load_channel_pairs_from_snirf(snirf_path)

    np.testing.assert_array_equal(pairings, [[1, 2], [3, 4]])


def test_validate_probe_settings_returns_canonical_registration_values() -> None:
    settings = validate_probe_settings(
        {
            "probe_settings": {
                "probe_units": "CM",
                "probe_orientation": "lia",
                "short_separation_flag": "DISTANCE",
                "short_separation_arg": 15.0,
                "embedding_step": 0.5,
                "max_embedding_steps": 1_000,
            }
        }
    )

    assert settings == {
        "probe_units": "cm",
        "probe_orientation": "LIA",
        "short_separation_flag": "distance",
        "short_separation_arg": 15.0,
        "embedding_step": 0.5,
        "max_embedding_steps": 1_000,
    }


def test_normalize_channel_pairings_returns_zero_based_indices() -> None:
    normalized = normalize_channel_pairings([[1, 1], [2, 2]], 2, 2, index_base="one")

    np.testing.assert_array_equal(normalized, [[0, 0], [1, 1]])


def test_flatten_channel_pairings_returns_unique_zero_based_matrix_rows() -> None:
    flattened = flatten_channel_pairings([[0, 0], [0, 0], [1, 0], [1, 1]], 2, 2)

    np.testing.assert_array_equal(flattened, [0, 2, 3])


def test_as_channel_pairing_array_rejects_invalid_shape_and_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        as_channel_pairing_array([[0, 0, 0]])
    with pytest.raises(ValueError, match="integer indices"):
        as_channel_pairing_array([[0.5, 0]])


def test_as_unit_direction_array_reports_unprepared_probe() -> None:
    np.testing.assert_array_equal(as_unit_direction_array([[0.0, 0.0, 1.0]], "directions"), [[0, 0, 1]])
    with pytest.raises(ValueError, match="prepare_probe may not have been run"):
        as_unit_direction_array([[0.0, 0.0, 2.0]], "directions")


def test_validate_prepared_probe_requires_zero_based_pairings() -> None:
    probe = {
        "sourcepos": [[0.0, 0.0, 0.0]],
        "detpos": [[1.0, 0.0, 0.0]],
        "sourcedir": [[1.0, 0.0, 0.0]],
        "detnorms": [[-1.0, 0.0, 0.0]],
        "source_elements": [0],
        "detector_elements": [0],
        "channel_pairings": [[0, 0]],
    }

    validated = validate_prepared_probe(probe, element_count=1)

    np.testing.assert_array_equal(validated["channel_pairings"], [[0, 0]])
    with pytest.raises(ValueError, match="out-of-range source"):
        validate_prepared_probe(probe | {"channel_pairings": [[1, 1]]}, element_count=1)
