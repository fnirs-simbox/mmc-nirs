import struct
from pathlib import Path

import numpy as np
import pytest

from mmcnirs.mmc.history import read_flux, read_history
from mmcnirs.mmc.photons import compute_detected_photon_weights


_HISTORY_HEADER = struct.Struct("<4s7IfIfi4I")


def _history_block(
    records: np.ndarray,
    *,
    medium_count: int,
    savedetflag: int,
    unitinmm: float = 1.0,
    seed_bytes: int = 0,
) -> bytes:
    photon_count, column_count = records.shape
    header = _HISTORY_HEADER.pack(
        b"MCXH",
        1,
        medium_count,
        2,
        column_count,
        100,
        photon_count,
        photon_count,
        unitinmm,
        seed_bytes,
        1.0,
        1,
        1,
        savedetflag,
        0,
        0,
    )
    seeds = bytes(photon_count * seed_bytes)
    return header + np.asarray(records, dtype="<f4").tobytes() + seeds


def test_read_history_preserves_weight_inputs(tmp_path: Path) -> None:
    # D, P, and W fields for two media.
    savedetflag = (1 << 0) | (1 << 2) | (1 << 6)
    records = np.array(
        [
            [1.0, 1.0, 2.0, 0.5],
            [2.0, 3.0, 4.0, 0.25],
        ]
    )
    history_path = tmp_path / "simulation.mch"
    history_path.write_bytes(
        _history_block(
            records,
            medium_count=2,
            savedetflag=savedetflag,
            unitinmm=2.0,
        )
    )

    detected_photons = read_history(history_path)

    np.testing.assert_array_equal(detected_photons["detid"], [1.0, 2.0])
    np.testing.assert_array_equal(
        detected_photons["ppath"],
        [[1.0, 2.0], [3.0, 4.0]],
    )
    np.testing.assert_array_equal(detected_photons["w0"], [0.5, 0.25])
    assert detected_photons["unitinmm"] == 2.0

    properties = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 1.0, 0.9, 1.4],
            [0.2, 1.0, 0.9, 1.4],
        ]
    )
    actual_weights = compute_detected_photon_weights(
        detected_photons,
        optical_properties=properties,
    )
    expected_weights = np.array([0.5, 0.25]) * np.exp(-np.array([1.0, 2.2]))
    np.testing.assert_allclose(
        actual_weights,
        expected_weights,
        rtol=1e-12,
        atol=1e-14,
    )


def test_read_history_concatenates_compatible_blocks(tmp_path: Path) -> None:
    savedetflag = (1 << 0) | (1 << 2)
    first = _history_block(
        np.array([[1.0, 2.0, 3.0]]),
        medium_count=2,
        savedetflag=savedetflag,
        seed_bytes=4,
    )
    second = _history_block(
        np.array([[2.0, 4.0, 5.0]]),
        medium_count=2,
        savedetflag=savedetflag,
        seed_bytes=4,
    )
    history_path = tmp_path / "simulation.mch"
    history_path.write_bytes(first + second)

    detected_photons = read_history(history_path)

    np.testing.assert_array_equal(detected_photons["detid"], [1.0, 2.0])
    np.testing.assert_array_equal(
        detected_photons["ppath"],
        [[2.0, 3.0], [4.0, 5.0]],
    )


def test_read_history_preserves_single_medium_vector_shape(tmp_path: Path) -> None:
    savedetflag = (1 << 0) | (1 << 2)
    history_path = tmp_path / "simulation.mch"
    history_path.write_bytes(
        _history_block(
            np.array([[1.0, 2.0], [2.0, 3.0]]),
            medium_count=1,
            savedetflag=savedetflag,
        )
    )

    detected_photons = read_history(history_path)

    np.testing.assert_array_equal(detected_photons["detid"], [1.0, 2.0])
    np.testing.assert_array_equal(detected_photons["ppath"], [[2.0], [3.0]])


def test_read_history_accepts_mmc_default_layout_when_savedetflag_is_zero(tmp_path: Path) -> None:
    # D, N, P, X, V, and W fields for one medium, as emitted by MMC 2.8.0.
    records = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.5],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.25],
        ]
    )
    history_path = tmp_path / "simulation.mch"
    history_path.write_bytes(_history_block(records, medium_count=1, savedetflag=0))

    detected_photons = read_history(history_path)

    np.testing.assert_array_equal(detected_photons["detid"], [1.0, 2.0])
    np.testing.assert_array_equal(detected_photons["nscat"], [[2.0], [3.0]])
    np.testing.assert_array_equal(detected_photons["ppath"], [[3.0], [4.0]])
    np.testing.assert_array_equal(detected_photons["p"], [[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]])
    np.testing.assert_array_equal(detected_photons["v"], [[7.0, 8.0, 9.0], [8.0, 9.0, 10.0]])
    np.testing.assert_array_equal(detected_photons["w0"], [0.5, 0.25])


def test_read_history_rejects_missing_partial_paths(tmp_path: Path) -> None:
    history_path = tmp_path / "simulation.mch"
    history_path.write_bytes(
        _history_block(
            np.array([[1.0]]),
            medium_count=2,
            savedetflag=1 << 0,
        )
    )

    with pytest.raises(ValueError, match="missing required fields: ppath"):
        read_history(history_path)


def test_read_flux_returns_second_column(tmp_path: Path) -> None:
    flux_path = tmp_path / "simulation.dat"
    flux_path.write_text("0 1.5\n1 2.5\n", encoding="utf-8")

    np.testing.assert_array_equal(read_flux(flux_path), [1.5, 2.5])
