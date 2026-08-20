import numpy as np
import pytest

from mmcnirs.mmc.photons import compute_detected_photon_weights


@pytest.fixture
def detected_photons() -> dict[str, np.ndarray]:
    return {
        "ppath": np.array(
            [
                [2.0, 3.0],
                [5.0, 7.0],
                [0.0, 11.0],
            ]
        )
    }


@pytest.fixture
def optical_properties() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 1.0, 0.9, 1.4],
            [0.02, 1.0, 0.9, 1.4],
        ]
    )


def test_detected_photon_weights(
    detected_photons: dict[str, np.ndarray],
    optical_properties: np.ndarray,
) -> None:
    actual = compute_detected_photon_weights(
        detected_photons,
        optical_properties=optical_properties,
    )

    expected = np.exp(-np.array([0.26, 0.64, 0.22]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_detected_photon_weights_apply_initial_weights(
    detected_photons: dict[str, np.ndarray],
    optical_properties: np.ndarray,
) -> None:
    initial_weights = np.array([0.5, 0.25, 0.75])
    detected_photons["w0"] = initial_weights

    actual = compute_detected_photon_weights(
        detected_photons,
        optical_properties=optical_properties,
    )

    expected = initial_weights * np.exp(-np.array([0.26, 0.64, 0.22]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_detected_photon_weights_apply_unitinmm(
    detected_photons: dict[str, np.ndarray],
    optical_properties: np.ndarray,
) -> None:
    detected_photons["unitinmm"] = 2.5

    actual = compute_detected_photon_weights(
        detected_photons,
        optical_properties=optical_properties,
    )

    expected = np.exp(-2.5 * np.array([0.26, 0.64, 0.22]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_detected_photon_weights_reject_media_mismatch(
    detected_photons: dict[str, np.ndarray],
    optical_properties: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="ppath describes 2 media"):
        compute_detected_photon_weights(
            detected_photons,
            optical_properties=optical_properties[:2],
        )
