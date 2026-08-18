"""Generate MMC-based fNIRS Jacobians from prepared inputs."""

# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mmc_nirs.light_transport.prepare_jacobian_inputs import prepare_jacobian_inputs
from mmc_nirs.mmc.history import read_cli_output, read_flux
from mmc_nirs.mmc.photons import compute_detected_photon_weights
from mmc_nirs.mmc.runner import run_mmc
from mmc_nirs.utils.jacobian_utils import (
    JACOBIAN_TSTEP_SECONDS,
    build_jacobian_mmc_config,
    load_jacobian_result,
    mmc_to_json,
    resolve_jacobian_save_path,
    save_jacobian_result,
    validate_mmc_flux,
)

__all__ = ["generate_jacobian"]

_DETECTOR_RADIUS_MM = 1.0


def _sum_detected_photon_weights(
    detected_photons: Mapping[str, ArrayLike],
    photon_weights: ArrayLike,
    detector_count: int,
) -> np.ndarray:
    """Sum detected-photon weights for every one-based MMC detector ID."""
    weights = np.asarray(photon_weights, dtype=float)
    detector_ids = np.asarray(detected_photons["detid"])
    if detector_ids.shape != weights.shape:
        raise ValueError("MMC history must contain one detector ID per detected-photon weight")
    if not np.all(np.isfinite(detector_ids)) or not np.all(detector_ids == np.floor(detector_ids)):
        raise ValueError("MMC history contains invalid detector IDs")
    detector_ids = detector_ids.astype(np.intp, copy=False)
    if detector_ids.size and (detector_ids.min() < 1 or detector_ids.max() > detector_count):
        raise ValueError("MMC history contains an out-of-range one-based detector ID")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Detected-photon weights must be finite and non-negative")
    return np.asarray(
        [weights[detector_ids == detector_index + 1].sum() for detector_index in range(detector_count)],
        dtype=float,
    )


def _calculate_jacobian(
    green_source: np.ndarray,
    green_detector: np.ndarray,
    green_source_detector: np.ndarray,
) -> np.ndarray:
    """Apply the legacy source-adjoint normalization for every optode pair."""
    source_count, node_count = green_source.shape
    detector_count = green_detector.shape[0]
    normalizers = np.asarray(green_source_detector, dtype=float).reshape(-1)
    if normalizers.shape != (source_count * detector_count,):
        raise ValueError("Green_sd must contain one value per source-detector pair")
    invalid_normalizers = ~np.isfinite(normalizers) | (normalizers <= 0)
    if np.any(invalid_normalizers):
        row = int(np.flatnonzero(invalid_normalizers)[0])
        source_index, detector_index = divmod(row, detector_count)
        raise ValueError(f"Green_sd must be finite and positive for source {source_index}, detector {detector_index}")

    jacobian = np.empty((source_count * detector_count, node_count), dtype=float)
    for source_index in range(source_count):
        for detector_index in range(detector_count):
            row = source_index * detector_count + detector_index
            jacobian[row] = green_source[source_index] * green_detector[detector_index] / normalizers[row]
    return jacobian


def generate_jacobian(
    prepared_mesh: Mapping[str, ArrayLike],
    prepared_probe: Mapping[str, ArrayLike],
    optical_properties: Mapping[str, Mapping[str, ArrayLike]],
    ordered_tissues: Sequence[str],
    mmc_settings: Mapping[str, Any],
    wavelength: str | int,
    save_path: str | Path | None,
    *,
    save: bool = True,
    overwrite: bool = False,
    timeout: float = 900,
) -> dict[str, np.ndarray]:
    """Generate a Jacobian for one wavelength from prepared mesh and probe data.

    Mesh/probe preparation and registration must already be complete. MMC runs
    in an isolated temporary directory. When saving is enabled, an existing
    compatible archive is returned without rerunning MMC unless ``overwrite``
    is true.

    ``prepared_mesh`` must contain canonical ``nodes``, zero-based ``elements``,
    and ``element_tissue_values`` arrays. ``prepared_probe`` must contain the
    registered positions, directions, zero-based containing-element indices,
    and channel pairings produced by :func:`prepare_jacobian_probe`.
    """
    if not isinstance(save, bool):
        raise TypeError("save must be a boolean")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")

    resolved_save_path = resolve_jacobian_save_path(save_path) if save else None
    if resolved_save_path is not None and resolved_save_path.is_file() and not overwrite:
        return load_jacobian_result(resolved_save_path)
    if isinstance(timeout, bool) or not isinstance(timeout, Real) or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")

    inputs = prepare_jacobian_inputs(
        prepared_mesh,
        prepared_probe,
        optical_properties,
        ordered_tissues,
        mmc_settings,
        wavelength,
    )
    base_config = build_jacobian_mmc_config(
        inputs.nodes,
        inputs.elements,
        inputs.element_tissue_values,
        inputs.selected_properties,
        inputs.photon_count,
    )
    detector_positions_with_radius = np.column_stack(
        (inputs.detector_positions, np.full(len(inputs.detector_positions), _DETECTOR_RADIUS_MM))
    )

    source_count = len(inputs.source_positions)
    detector_count = len(inputs.detector_positions)
    node_count = len(inputs.nodes)
    row_count = source_count * detector_count
    green_source = np.zeros((source_count, node_count), dtype=float)
    green_detector = np.zeros((detector_count, node_count), dtype=float)
    green_source_detector = np.zeros((row_count, 1), dtype=float)
    measurements_zero = np.zeros((row_count, 1), dtype=float)

    with TemporaryDirectory(prefix="mmc-nirs-jacobian-") as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)

        for source_index in range(source_count):
            output_stub = temporary_directory / f"source_{source_index:04d}"
            source_config = base_config | {
                "srcpos": inputs.source_positions[source_index].tolist(),
                "e0": int(inputs.source_elements[source_index]) + 1,
                "srcdir": inputs.source_directions[source_index].tolist(),
                "detpos": detector_positions_with_radius.tolist(),
            }
            config_path = output_stub.with_suffix(".json")
            mmc_to_json(source_config, config_path)
            run_mmc(config_path, working_directory=temporary_directory, timeout=float(timeout))

            source_flux, detected_photons = read_cli_output(output_stub)
            source_flux = validate_mmc_flux(source_flux, node_count, f"source {source_index}")
            green_source[source_index] = source_flux * JACOBIAN_TSTEP_SECONDS
            photon_weights = compute_detected_photon_weights(
                detected_photons,
                optical_properties=inputs.selected_properties,
            )
            detector_weight_sums = _sum_detected_photon_weights(
                detected_photons,
                photon_weights,
                detector_count,
            )
            row_start = source_index * detector_count
            row_stop = row_start + detector_count
            measurements_zero[row_start:row_stop, 0] = detector_weight_sums
            green_source_detector[row_start:row_stop, 0] = (
                source_flux[inputs.closest_detector_nodes] * JACOBIAN_TSTEP_SECONDS
            )

        for detector_index in range(detector_count):
            output_stub = temporary_directory / f"detector_{detector_index:04d}"
            detector_config = base_config | {
                "srcpos": inputs.detector_positions[detector_index].tolist(),
                "e0": int(inputs.detector_elements[detector_index]) + 1,
                "srcdir": inputs.detector_directions[detector_index].tolist(),
            }
            config_path = output_stub.with_suffix(".json")
            mmc_to_json(detector_config, config_path)
            run_mmc(config_path, working_directory=temporary_directory, timeout=float(timeout))

            detector_flux = read_flux(output_stub.with_suffix(".dat"))
            green_detector[detector_index] = (
                validate_mmc_flux(detector_flux, node_count, f"detector {detector_index}") * JACOBIAN_TSTEP_SECONDS
            )

    result = {
        "Green_d": green_detector,
        "Green_s": green_source,
        "Green_sd": green_source_detector,
        "J": _calculate_jacobian(green_source, green_detector, green_source_detector),
        "channelidx": inputs.channel_indices,
        "mea0": measurements_zero,
        "sourcepos": inputs.source_positions,
        "detpos": detector_positions_with_radius,
        "detnorms": inputs.detector_directions,
        "sourcedir": inputs.source_directions,
    }
    if resolved_save_path is not None:
        save_jacobian_result(resolved_save_path, result)
    return result
