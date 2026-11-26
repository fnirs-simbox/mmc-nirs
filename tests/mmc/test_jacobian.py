import json
import struct
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import mmcnirs.mmc.jacobian as jacobian_module
from mmcnirs.mmc.jacobian import generate_jacobian

_HISTORY_HEADER = struct.Struct("<4s7IfIfi4I")
_RESULT_KEYS = {
    "Green_d",
    "Green_s",
    "Green_sd",
    "J",
    "channelidx",
    "mea0",
    "sourcepos",
    "detpos",
    "detnorms",
    "sourcedir",
}


@pytest.fixture(autouse=True)
def runtime_lookup(monkeypatch):
    lookup = Mock(return_value=Path("/mock/mmc"))
    monkeypatch.setattr(jacobian_module, "get_mmc_executable", lookup)
    return lookup


def _write_flux(path: Path, values: np.ndarray) -> None:
    lines = [f"{index} {value}\n" for index, value in enumerate(values)]
    path.write_text("".join(lines), encoding="utf-8")


def _write_history(path: Path, records: np.ndarray, detector_count: int = 2) -> None:
    records = np.asarray(records, dtype="<f4").reshape(-1, 2)
    records = np.column_stack((records, np.ones(len(records), dtype="<f4")))
    photon_count = len(records)
    savedetflag = (1 << 0) | (1 << 2) | (1 << 6)  # Detector ID, partial path, initial weight.
    header = _HISTORY_HEADER.pack(
        b"MCXH",
        1,
        1,
        detector_count,
        3,
        100,
        photon_count,
        photon_count,
        1.0,
        0,
        1.0,
        1,
        1,
        savedetflag,
        0,
        0,
    )
    path.write_bytes(header + records.tobytes())


def _mock_mmc_outputs(monkeypatch, *, zero_source_flux: bool = False):
    source_fluxes = {
        "source_0000": np.array([0.0 if zero_source_flux else 2.0, 4.0, 6.0, 8.0]),
        "source_0001": np.array([3.0, 9.0, 12.0, 6.0]),
    }
    detector_fluxes = {
        "detector_0000": np.array([5.0, 10.0, 15.0, 20.0]),
        "detector_0001": np.array([7.0, 14.0, 21.0, 28.0]),
    }
    histories = {
        "source_0000": np.array([[1.0, 1.0], [2.0, 2.0], [2.0, 3.0]]),
        "source_0001": np.array([[1.0, 4.0], [2.0, 5.0]]),
    }
    sessions = []

    def fake_run_mmc(config_path, *, working_directory, timeout, extra_args, executable):
        assert executable == Path("/mock/mmc")
        config_path = Path(config_path)
        working_directory = Path(working_directory)
        session = json.loads(config_path.read_text(encoding="utf-8"))["Session"]
        assert session["Photons"] == 100
        session_id = session["ID"]
        if session_id.startswith("source_"):
            assert extra_args[-2:] == ["-H", "10000000"]
        else:
            assert "-H" not in extra_args
        sessions.append(session_id)
        output_stub = working_directory / session_id
        if session_id in source_fluxes:
            _write_flux(output_stub.with_suffix(".dat"), source_fluxes[session_id])
            _write_history(output_stub.with_suffix(".mch"), histories[session_id])
        else:
            _write_flux(output_stub.with_suffix(".dat"), detector_fluxes[session_id])

    monkeypatch.setattr(jacobian_module, "run_mmc", fake_run_mmc)
    return sessions


@pytest.mark.parametrize("saved", [10_000_000, 10_000_001])
def test_complete_history_rejects_capacity_even_when_header_counts_match(saved):
    history = {"detected_counts": np.array([saved]), "saved_counts": np.array([saved])}
    with pytest.raises(RuntimeError, match="recording capacity for source 7"):
        jacobian_module._validate_complete_detected_history(history, 7)


def test_complete_history_accepts_below_capacity():
    history = {"detected_counts": np.array([9_999_999]), "saved_counts": np.array([9_999_999])}
    jacobian_module._validate_complete_detected_history(history, 7)


def test_generate_jacobian_reuses_an_existing_saved_result(tmp_path: Path, monkeypatch, runtime_lookup) -> None:
    save_path = tmp_path / "jacobian_690.npz"
    cached = {key: np.array([index]) for index, key in enumerate(sorted(_RESULT_KEYS))}
    np.savez(save_path, **cached)
    monkeypatch.setattr(jacobian_module, "run_mmc", lambda *args, **kwargs: pytest.fail("MMC ran"))

    result = generate_jacobian({}, {}, {}, {}, 690, save_path)

    runtime_lookup.assert_not_called()
    for key, value in cached.items():
        np.testing.assert_array_equal(result[key], value)


def test_generate_jacobian_rejects_an_incompatible_saved_result(tmp_path: Path) -> None:
    save_path = tmp_path / "jacobian_690.npz"
    np.savez(save_path, J=np.ones((1, 1)))

    with pytest.raises(ValueError, match="missing required field"):
        generate_jacobian({}, {}, {}, {}, 690, save_path)


def test_generate_jacobian_overwrites_archive_with_normalized_values(tmp_path: Path, monkeypatch, jacobian_inputs):
    sessions = _mock_mmc_outputs(monkeypatch)
    save_path = tmp_path / "jacobian_690.npz"
    cached = {key: np.array([-1]) for key in _RESULT_KEYS}
    np.savez(save_path, **cached)

    result = generate_jacobian(*jacobian_inputs, {"nphoton": 100}, 690, save_path, overwrite=True)

    assert sessions == ["source_0000", "source_0001", "detector_0000", "detector_0001"]
    np.testing.assert_allclose(result["Green_sd"][:, 0], [2.0, 4.25, 3.0, 9.0])
    # The second detector interpolates the source fluence to 9. Each node has volume 1/24 mm³.
    np.testing.assert_allclose(result["J"][3], -np.array([21.0, 126.0, 252.0, 168.0]) / (24 * 9))
    weights = np.exp(-0.1 * np.arange(1, 6))
    expected_measurements = np.array([weights[0], weights[1:3].sum(), weights[3], weights[4]]) / 100
    np.testing.assert_allclose(result["mea0"][:, 0], expected_measurements)
    with np.load(save_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["J"], result["J"])


def test_generate_jacobian_resolves_runtime_once_for_all_runs(
    monkeypatch,
    runtime_lookup,
    jacobian_inputs,
) -> None:
    _mock_mmc_outputs(monkeypatch)

    result = generate_jacobian(
        *jacobian_inputs,
        {"nphoton": 100},
        690,
        None,
        save=False,
        version="v2026",
        update_manifest=True,
    )

    assert result["J"].shape == (4, 4)
    runtime_lookup.assert_called_once_with(version="v2026", update_manifest=True)


def test_generate_jacobian_accepts_zero_source_flux(
    tmp_path: Path,
    monkeypatch,
    jacobian_inputs,
) -> None:
    _mock_mmc_outputs(monkeypatch, zero_source_flux=True)

    result = generate_jacobian(
        *jacobian_inputs,
        {"nphoton": 100},
        690,
        tmp_path / "jacobian.npz",
    )

    np.testing.assert_array_equal(result["J"][:2, 0], [0.0, 0.0])


@pytest.mark.parametrize(
    ("mmc_settings", "message"),
    [
        ({}, "missing required field"),
        ({"nphoton": 0}, "positive integer"),
        ({"nphoton": 1.5}, "positive integer"),
        ({"nphoton": 2_147_483_648}, "at most 2,147,483,647 photons"),
        ({"nphoton": 100, "tstep": 1.0}, "unsupported field"),
    ],
)
def test_generate_jacobian_rejects_invalid_mmc_settings(
    tmp_path: Path,
    jacobian_inputs,
    mmc_settings,
    message,
    runtime_lookup,
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_jacobian(
            *jacobian_inputs,
            mmc_settings,
            690,
            tmp_path / "jacobian.npz",
        )
    runtime_lookup.assert_not_called()


def test_generate_jacobian_requires_mmc_outputs(
    tmp_path: Path,
    monkeypatch,
    jacobian_inputs,
) -> None:
    monkeypatch.setattr(jacobian_module, "run_mmc", lambda *args, **kwargs: None)

    with pytest.raises(FileNotFoundError, match="source_0000.dat"):
        generate_jacobian(
            *jacobian_inputs,
            {"nphoton": 100},
            690,
            tmp_path / "jacobian.npz",
        )


@pytest.mark.parametrize(
    ("detector_ids", "weights", "message"),
    [
        ([0], [1.0], "out-of-range"),
        ([3], [1.0], "out-of-range"),
        ([np.nan], [1.0], "invalid detector IDs"),
        ([1], [-1.0], "finite and non-negative"),
    ],
)
def test_detected_photon_weight_sums_reject_invalid_history(detector_ids, weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        jacobian_module._sum_detected_photon_weights(
            {"detid": np.asarray(detector_ids)},
            np.asarray(weights),
            detector_count=2,
        )
