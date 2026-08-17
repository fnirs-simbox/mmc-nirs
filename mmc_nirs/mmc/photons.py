"""MMC detected-photon weight calculations."""

# SPDX-License-Identifier: MIT

import numpy as np


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

    if "ppath" not in detected_photons:
        raise ValueError("detected_photons must contain 'ppath'")

    if optical_properties is None:
        try:
            optical_properties = detected_photons["prop"]
        except KeyError as exc:
            raise ValueError(
                "optical_properties must be provided when detected_photons does not contain 'prop'"
            ) from exc

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

    if unitinmm is None:
        unitinmm = detected_photons.get("unitinmm", 1.0)

    unitinmm = float(unitinmm)
    if not np.isfinite(unitinmm) or unitinmm <= 0:
        raise ValueError("unitinmm must be a finite positive number")

    # Row 0 is the background medium and is not represented in ppath.
    absorption_coefficients = properties[1:, 0]

    # Convert stored path lengths to millimeters exactly once here, then
    # accumulate Beer-Lambert attenuation for each detected photon.
    optical_depth = (ppath @ absorption_coefficients) * unitinmm

    initial_weights = detected_photons.get("w0")
    if initial_weights is None:
        initial_weights = np.ones(ppath.shape[0], dtype=float)
    else:
        initial_weights = np.asarray(initial_weights, dtype=float)
        if initial_weights.shape != (ppath.shape[0],):
            raise ValueError("w0 must contain one initial weight per detected photon")

    return initial_weights * np.exp(-optical_depth)
