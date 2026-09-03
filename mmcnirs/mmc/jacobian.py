"""Generate MMC-based fNIRS Jacobians from prepared inputs."""

# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from mmcnirs.light_transport.prepare_jacobian_inputs import prepare_jacobian_inputs
from mmcnirs.mmc.history import read_cli_output, read_flux
from mmcnirs.mmc.photons import compute_detected_photon_weights
from mmcnirs.mmc.runner import run_mmc
from mmcnirs.utils.jacobian_utils import (
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
_DETECTOR_AREA_MM2 = np.pi * _DETECTOR_RADIUS_MM**2
_ALPHA = 8.47


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


def _calculate_node_volumes(
    nodes: ArrayLike,
    elements: ArrayLike,
) -> np.ndarray:
    """Calculate barycentric dual volumes for tetrahedral mesh nodes."""
    node_array = np.asarray(nodes, dtype=float)
    element_array = np.asarray(elements)

    if node_array.ndim != 2 or node_array.shape[1] != 3:
        raise ValueError("Mesh nodes must have shape (N, 3)")
    if element_array.ndim != 2 or element_array.shape[1] != 4:
        raise ValueError("Mesh elements must have shape (E, 4)")
    if not np.issubdtype(element_array.dtype, np.integer):
        if not np.all(np.isfinite(element_array)) or not np.all(element_array == np.floor(element_array)):
            raise ValueError("Mesh elements must contain integer node indices")
        element_array = element_array.astype(np.intp)
    else:
        element_array = element_array.astype(np.intp, copy=False)

    node_count = len(node_array)
    if element_array.size and (element_array.min() < 0 or element_array.max() >= node_count):
        raise ValueError("Mesh elements contain out-of-range node indices")

    # tetrahedral points
    p0 = node_array[element_array[:, 0]]
    p1 = node_array[element_array[:, 1]]
    p2 = node_array[element_array[:, 2]]
    p3 = node_array[element_array[:, 3]]

    # tetrahedral volumes
    element_volumes = (
        np.abs(
            np.einsum(
                "ij,ij->i",
                p1 - p0,
                np.cross(p2 - p0, p3 - p0),
            )
        )
        / 6.0
    )

    if not np.all(np.isfinite(element_volumes)) or np.any(element_volumes <= 0):
        raise ValueError("Mesh contains invalid or degenerate tetrahedra")

    # assign volume to each node depending on it's tetrahedras
    node_volumes = np.zeros(node_count, dtype=float)
    for local_node_index in range(4):
        np.add.at(
            node_volumes,
            element_array[:, local_node_index],
            element_volumes / 4.0,
        )

    if np.any(node_volumes <= 0):
        raise ValueError("Every mesh node must have a positive associated volume")

    return node_volumes


def _calculate_jacobian(
    green_source: np.ndarray,
    green_detector: np.ndarray,
    green_source_detector: np.ndarray,
    node_volumes: np.ndarray,
) -> np.ndarray:
    """Calculate the Rytov log-intensity Jacobian for every source-detector pair."""
    source_count, node_count = green_source.shape
    detector_count = green_detector.shape[0]

    volumes = np.asarray(node_volumes, dtype=float).reshape(-1)

    normalizers = np.asarray(green_source_detector, dtype=float).reshape(-1)
    if normalizers.shape != (source_count * detector_count,):
        raise ValueError("Green_sd must contain one value per source-detector pair")
    invalid_normalizers = ~np.isfinite(normalizers) | (normalizers < 0)
    if np.any(invalid_normalizers):
        row = int(np.flatnonzero(invalid_normalizers)[0])
        source_index, detector_index = divmod(row, detector_count)
        raise ValueError(f"Green_sd must be finite and positive for source {source_index}, detector {detector_index}")

    jacobian = np.empty((source_count * detector_count, node_count), dtype=float)
    for source_index in range(source_count):
        for detector_index in range(detector_count):
            row = source_index * detector_count + detector_index

            if normalizers[row] == 0:
                jacobian[row] = 0.0
                continue

            jacobian[row] = -volumes * green_source[source_index] * green_detector[detector_index] / normalizers[row]
    return jacobian


def generate_jacobian(
    prepared_mesh: Mapping[str, ArrayLike],
    prepared_probe: Mapping[str, ArrayLike],
    optical_properties: Mapping[str, Mapping[str, ArrayLike]],
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
    positional ``element_tissue_ids``, and parallel ``ordered_tissue_ids`` and
    ``ordered_tissues`` arrays. ``prepared_probe`` must contain the registered
    positions, directions, zero-based containing-element indices, and channel
    pairings produced by :func:`prepare_probe`.
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
        mmc_settings,
        wavelength,
    )
    base_config = build_jacobian_mmc_config(
        inputs.nodes,
        inputs.elements,
        inputs.element_tissue_ids,
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

    node_volumes = _calculate_node_volumes(inputs.nodes, inputs.elements)

    green_source = np.zeros((source_count, node_count), dtype=float)
    green_detector = np.zeros((detector_count, node_count), dtype=float)
    green_source_detector = np.zeros((row_count, 1), dtype=float)
    measurements_zero = np.zeros((row_count, 1), dtype=float)

    with TemporaryDirectory(prefix="mmcnirs-jacobian-") as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)

        source_progress = tqdm(range(source_count), desc="MMC sources", unit="source")
        for source_index in source_progress:
            source_progress.set_postfix_str(f"source {source_index}")
            output_stub = temporary_directory / f"source_{source_index:04d}"
            source_config = base_config | {
                "srcpos": inputs.source_positions[source_index].tolist(),
                "e0": int(inputs.source_elements[source_index]) + 1,
                "srcdir": inputs.source_directions[source_index].tolist(),
                "detpos": detector_positions_with_radius.tolist(),
            }
            config_path = output_stub.with_suffix(".json")
            mmc_to_json(source_config, config_path)
            try:
                run_mmc(config_path, working_directory=temporary_directory, timeout=float(timeout))
            except TimeoutError as error:
                raise TimeoutError(
                    f"MMC failed while computing source {source_index} ({source_index + 1}/{source_count}): {error}"
                ) from error

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

            measurements_zero[row_start:row_stop, 0] = detector_weight_sums / inputs.photon_count

            detector_ids = np.asarray(detected_photons["detid"], dtype=int)
            for detector_index in np.flatnonzero(detector_weight_sums == 0):
                detected_count = np.sum(detector_ids == detector_index + 1)
                print(
                    f"source {source_index}, detector {detector_index}: "
                    f"detected_count={detected_count}, "
                    f"weight_sum={detector_weight_sums[detector_index]}"
                )

            # Baseline source-detector diffuse reflectance used for Rytov normalization.
            green_source_detector[row_start:row_stop, 0] = (
                detector_weight_sums / _DETECTOR_AREA_MM2 / inputs.photon_count
            )

        detector_progress = tqdm(range(detector_count), desc="MMC detectors", unit="detector")
        for detector_index in detector_progress:
            detector_progress.set_postfix_str(f"detector {detector_index}")
            output_stub = temporary_directory / f"detector_{detector_index:04d}"
            detector_config = base_config | {
                "srcpos": inputs.detector_positions[detector_index].tolist(),
                "e0": int(inputs.detector_elements[detector_index]) + 1,
                "srcdir": inputs.detector_directions[detector_index].tolist(),
            }
            config_path = output_stub.with_suffix(".json")
            mmc_to_json(detector_config, config_path)
            try:
                run_mmc(config_path, working_directory=temporary_directory, timeout=float(timeout))
            except TimeoutError as error:
                raise TimeoutError(
                    f"MMC failed while computing detector {detector_index} "
                    f"({detector_index + 1}/{detector_count}): {error}"
                ) from error

            detector_flux = read_flux(output_stub.with_suffix(".dat"))
            detector_flux = validate_mmc_flux(detector_flux, node_count, f"detector {detector_index}")
            green_detector[detector_index] = detector_flux * JACOBIAN_TSTEP_SECONDS

    result = {
        "Green_d": green_detector,
        "Green_s": green_source,
        "Green_sd": green_source_detector,
        "J": _calculate_jacobian(green_source, green_detector, green_source_detector, node_volumes),
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
