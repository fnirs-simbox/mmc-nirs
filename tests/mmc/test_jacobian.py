import json
import struct
from pathlib import Path

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


@pytest.fixture
def prepared_mesh() -> dict[str, np.ndarray]:
    return {
        "nodes": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        "elements": np.array([[0, 1, 2, 3]]),
        "element_tissue_ids": np.array([1]),
        "ordered_tissue_ids": np.array([0, 1]),
        "ordered_tissues": np.array(["ambient_air", "tissue"]),
    }


@pytest.fixture
def prepared_probe() -> dict[str, np.ndarray]:
    return {
        "sourcepos": np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]]),
        "detpos": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "sourcedir": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "detnorms": np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]),
        "source_elements": np.array([0, 0]),
        "detector_elements": np.array([0, 0]),
        "channel_pairings": np.array([[0, 0], [0, 0], [1, 0], [1, 1]]),
    }


@pytest.fixture
def optical_properties() -> dict[str, dict[str, list[float]]]:
    return {
        "690": {
            "ambient_air": [0.0, 0.0, 1.0, 1.0],
            "tissue": [0.1, 1.0, 0.9, 1.4],
        }
    }


def _write_flux(path: Path, values: np.ndarray) -> None:
    lines = [f"{index} {value}\n" for index, value in enumerate(values)]
    path.write_text("".join(lines), encoding="utf-8")


def _write_history(path: Path, records: np.ndarray, detector_count: int = 2) -> None:
    records = np.asarray(records, dtype="<f4").reshape(-1, 2)
    photon_count = len(records)
    savedetflag = (1 << 0) | (1 << 2)  # Detector ID and partial path.
    header = _HISTORY_HEADER.pack(
        b"MCXH",
        1,
        1,
        detector_count,
        2,
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


def _mock_mmc_outputs(monkeypatch, *, zero_normalizer: bool = False):
    source_fluxes = {
        "source_0000": np.array([0.0 if zero_normalizer else 2.0, 4.0, 6.0, 8.0]),
        "source_0001": np.array([3.0, 6.0, 9.0, 12.0]),
    }
    detector_fluxes = {
        "detector_0000": np.array([5.0, 10.0, 15.0, 20.0]),
        "detector_0001": np.array([7.0, 14.0, 21.0, 28.0]),
    }
    histories = {
        "source_0000": np.array([[1.0, 1.0], [2.0, 2.0], [2.0, 3.0]]),
        "source_0001": np.array([[1.0, 4.0]]),
    }
    calls = []

    def fake_run_mmc(config_path, *, working_directory, timeout):
        config_path = Path(config_path)
        working_directory = Path(working_directory)
        document = json.loads(config_path.read_text(encoding="utf-8"))
        session_id = document["Session"]["ID"]
        calls.append((document, working_directory, timeout))
        output_stub = working_directory / session_id
        if session_id in source_fluxes:
            _write_flux(output_stub.with_suffix(".dat"), source_fluxes[session_id])
            _write_history(output_stub.with_suffix(".mch"), histories[session_id])
        else:
            _write_flux(output_stub.with_suffix(".dat"), detector_fluxes[session_id])

    monkeypatch.setattr(jacobian_module, "run_mmc", fake_run_mmc)
    return calls, source_fluxes, detector_fluxes


def test_generate_jacobian_runs_mmc_and_preserves_legacy_calculations(
    tmp_path: Path,
    monkeypatch,
    prepared_mesh,
    prepared_probe,
    optical_properties,
) -> None:
    calls, source_fluxes, detector_fluxes = _mock_mmc_outputs(monkeypatch)
    save_path = tmp_path / "outputs" / "jacobian_690.npz"

    result = generate_jacobian(
        prepared_mesh,
        prepared_probe,
        optical_properties,
        {"nphoton": 100},
        690,
        save_path,
        timeout=12,
    )

    tstep = 5e-9
    expected_green_s = np.vstack((source_fluxes["source_0000"], source_fluxes["source_0001"])) * tstep
    expected_green_d = np.vstack((detector_fluxes["detector_0000"], detector_fluxes["detector_0001"])) * tstep
    expected_green_sd = np.array([[2.0], [4.0], [3.0], [6.0]]) * tstep
    expected_jacobian = np.vstack(
        [
            expected_green_s[0] * expected_green_d[0] / expected_green_sd[0],
            expected_green_s[0] * expected_green_d[1] / expected_green_sd[1],
            expected_green_s[1] * expected_green_d[0] / expected_green_sd[2],
            expected_green_s[1] * expected_green_d[1] / expected_green_sd[3],
        ]
    )
    expected_measurements = np.array(
        [
            [np.exp(-0.1)],
            [np.exp(-0.2) + np.exp(-0.3)],
            [np.exp(-0.4)],
            [0.0],
        ]
    )

    np.testing.assert_allclose(result["Green_s"], expected_green_s)
    np.testing.assert_allclose(result["Green_d"], expected_green_d)
    np.testing.assert_allclose(result["Green_sd"], expected_green_sd)
    np.testing.assert_allclose(result["J"], expected_jacobian)
    np.testing.assert_allclose(result["mea0"], expected_measurements, rtol=1e-6)
    np.testing.assert_array_equal(result["channelidx"], [0, 2, 3])
    np.testing.assert_array_equal(result["detpos"][:, 3], [1.0, 1.0])
    assert save_path.is_file()

    assert len(calls) == 4
    assert all(timeout == 12 for _, _, timeout in calls)
    working_directories = {working_directory for _, working_directory, _ in calls}
    assert len(working_directories) == 1
    assert not next(iter(working_directories)).exists()

    source_documents = [document for document, _, _ in calls[:2]]
    detector_documents = [document for document, _, _ in calls[2:]]
    assert [document["Mesh"]["InitElem"] for document in source_documents] == [1, 1]
    assert [document["Mesh"]["InitElem"] for document in detector_documents] == [1, 1]
    assert all(document["Session"]["Photons"] == 100 for document, _, _ in calls)
    assert all(document["Mesh"]["MeshElem"] == [[1, 2, 3, 4, 1]] for document, _, _ in calls)
    assert all(document["Forward"] == {"T0": 0.0, "T1": 5e-9, "Dt": 5e-9} for document, _, _ in calls)
    assert all(len(document["Optode"]["Detector"]) == 2 for document in source_documents)
    assert all(detector["R"] == 1.0 for document in source_documents for detector in document["Optode"]["Detector"])
    assert all("Detector" not in document["Optode"] for document in detector_documents)


def test_generate_jacobian_reuses_an_existing_saved_result(tmp_path: Path, monkeypatch) -> None:
    save_path = tmp_path / "jacobian_690.npz"
    cached = {key: np.array([index]) for index, key in enumerate(sorted(_RESULT_KEYS))}
    np.savez(save_path, **cached)
    monkeypatch.setattr(jacobian_module, "run_mmc", lambda *args, **kwargs: pytest.fail("MMC ran"))

    result = generate_jacobian({}, {}, {}, {}, 690, save_path)

    for key, value in cached.items():
        np.testing.assert_array_equal(result[key], value)


def test_generate_jacobian_rejects_an_incompatible_saved_result(tmp_path: Path) -> None:
    save_path = tmp_path / "jacobian_690.npz"
    np.savez(save_path, J=np.ones((1, 1)))

    with pytest.raises(ValueError, match="missing required field"):
        generate_jacobian({}, {}, {}, {}, 690, save_path)


def test_generate_jacobian_overwrites_an_existing_result(
    tmp_path: Path,
    monkeypatch,
    prepared_mesh,
    prepared_probe,
    optical_properties,
) -> None:
    calls, _, _ = _mock_mmc_outputs(monkeypatch)
    save_path = tmp_path / "jacobian_690.npz"
    cached = {key: np.array([-1]) for key in _RESULT_KEYS}
    np.savez(save_path, **cached)

    result = generate_jacobian(
        prepared_mesh,
        prepared_probe,
        optical_properties,
        {"nphoton": 100},
        690,
        save_path,
        overwrite=True,
    )

    assert len(calls) == 4
    assert result["J"].shape == (4, 4)
    with np.load(save_path, allow_pickle=False) as archive:
        assert archive["J"].shape == (4, 4)


def test_generate_jacobian_can_skip_saving(
    monkeypatch,
    prepared_mesh,
    prepared_probe,
    optical_properties,
) -> None:
    calls, _, _ = _mock_mmc_outputs(monkeypatch)

    result = generate_jacobian(
        prepared_mesh,
        prepared_probe,
        optical_properties,
        {"nphoton": 100},
        690,
        None,
        save=False,
    )

    assert len(calls) == 4
    assert result["J"].shape == (4, 4)


def test_generate_jacobian_rejects_zero_source_detector_normalization(
    tmp_path: Path,
    monkeypatch,
    prepared_mesh,
    prepared_probe,
    optical_properties,
) -> None:
    _mock_mmc_outputs(monkeypatch, zero_normalizer=True)

    with pytest.raises(ValueError, match="Green_sd must be finite and positive for source 0, detector 0"):
        generate_jacobian(
            prepared_mesh,
            prepared_probe,
            optical_properties,
            {"nphoton": 100},
            690,
            tmp_path / "jacobian.npz",
        )


@pytest.mark.parametrize(
    ("mmc_settings", "error", "message"),
    [
        ({}, ValueError, "missing required field"),
        ({"nphoton": 0}, ValueError, "positive integer"),
        ({"nphoton": 1.5}, ValueError, "positive integer"),
        ({"nphoton": 100, "tstep": 1.0}, ValueError, "unsupported field"),
    ],
)
def test_generate_jacobian_rejects_invalid_mmc_settings(
    tmp_path: Path,
    prepared_mesh,
    prepared_probe,
    optical_properties,
    mmc_settings,
    error,
    message,
) -> None:
    with pytest.raises(error, match=message):
        generate_jacobian(
            prepared_mesh,
            prepared_probe,
            optical_properties,
            mmc_settings,
            690,
            tmp_path / "jacobian.npz",
        )


def test_generate_jacobian_requires_mmc_outputs(
    tmp_path: Path,
    monkeypatch,
    prepared_mesh,
    prepared_probe,
    optical_properties,
) -> None:
    monkeypatch.setattr(jacobian_module, "run_mmc", lambda *args, **kwargs: None)

    with pytest.raises(FileNotFoundError, match="source_0000.dat"):
        generate_jacobian(
            prepared_mesh,
            prepared_probe,
            optical_properties,
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
