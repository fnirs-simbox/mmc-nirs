"""MMC detected-photon weight calculations."""

# SPDX-License-Identifier: MIT

import numpy as np
import warnings


def compute_detected_photon_weights(
    detected_photons: dict,
    optical_properties=None,
    unitinmm=None,
) -> np.ndarray:
    """Compute detected-photon weights from absorption along photon paths.

    Parameters
    ----------
    detected_photons : dict
        Detected-photon data containing ``ppath`` with shape
        ``(n_photons, n_media)``. The dictionary may also contain ``w0`` and
        ``unitinmm`` as returned by ``read_history``.
    optical_properties : array-like, optional
        Optical property table. Row 0 is the background medium and the
        remaining rows correspond to the media represented by ``ppath``.
        Column 0 contains absorption coefficients. If omitted, the function
        falls back to ``detected_photons["prop"]``.
    unitinmm : float, optional
        Millimeters per stored path-length unit. If omitted, defaults to
        ``detected_photons["unitinmm"]`` or 1.0.

    Returns
    -------
    numpy.ndarray
        Detected-photon weights with shape ``(n_photons,)``.

    Raises
    ------
    TypeError
        If ``detected_photons`` is not a dictionary.
    ValueError
        If required data are missing or array shapes are inconsistent.
    """
    if not isinstance(detected_photons, dict):
        raise TypeError("detected_photons must be a dictionary")

    if "w0" not in detected_photons:
        raise ValueError("detected_photons must contain 'w0'")

    if not np.allclose(detected_photons["w0"], 1.0, rtol=0.0, atol=1e-7):
        warnings.warn("Detected photons have non-unit launch weights; verify source configuration.")

    if "ppath" not in detected_photons:
        raise ValueError("detected_photons must contain 'ppath'")

    if optical_properties is None:
        try:
            optical_properties = detected_photons["prop"]
        except KeyError as exc:
            raise ValueError(
                "optical_properties must be provided when detected_photons does not contain 'prop'"
            ) from exc

    if unitinmm is None:
        if "unitinmm" not in detected_photons:
            raise ValueError("detected_photons must contain 'unitinmm'")
        unitinmm = detected_photons["unitinmm"]
    unitinmm = float(unitinmm)
    if not np.isfinite(unitinmm) or unitinmm <= 0:
        raise ValueError("unitinmm must be a finite positive number")

    ppath = np.asarray(detected_photons["ppath"], dtype=float)
    properties = np.asarray(optical_properties, dtype=float)

    if ppath.ndim != 2:
        raise ValueError("ppath must be a two-dimensional array")

    if properties.ndim != 2:
        raise ValueError("optical_properties must be a two-dimensional array")

    if properties.shape[1] < 1:
        raise ValueError("optical_properties must contain an absorption column")

    if properties.shape[0] < 2:
        raise ValueError("optical_properties must contain a background row and at least one tissue medium")

    n_media = properties.shape[0] - 1
    if ppath.shape[1] != n_media:
        raise ValueError(f"ppath describes {ppath.shape[1]} media, but optical_properties describes {n_media}")

    # Row 0 is the background medium and is not represented in ppath.
    absorption_coefficients = properties[1:, 0]

    # Convert stored path lengths to millimeters exactly once here, then
    # accumulate Beer-Lambert attenuation for each detected photon.
    optical_depth = (ppath @ absorption_coefficients) * unitinmm

    try:
        initial_weights = np.asarray(detected_photons["w0"], dtype=float)
    except KeyError as exc:
        raise ValueError(
            "detected_photons must contain 'w0'; "
            "MMC detected-photon history is expected to save the initial packet weight"
        ) from exc

    if initial_weights.shape != (ppath.shape[0],):
        raise ValueError("w0 must contain one initial weight per detected photon")
    if not np.all(np.isfinite(initial_weights)):
        raise ValueError("w0 must contain only finite values")

    return initial_weights * np.exp(-optical_depth)
