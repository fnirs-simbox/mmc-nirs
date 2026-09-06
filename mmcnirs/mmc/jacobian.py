"""Generate MMC-based fNIRS Jacobians from prepared inputs."""

# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from mmcnirs.light_transport.prepare_jacobian_inputs import prepare_jacobian_inputs
from mmcnirs.mmc.history import read_cli_output, read_flux
from mmcnirs.mmc.photons import compute_detected_photon_weights
from mmcnirs.mmc.runner import run_mmc
from mmcnirs.utils.jacobian_utils import (
    build_jacobian_mmc_config,
    load_jacobian_result,
    mmc_to_json,
    resolve_jacobian_save_path,
    save_jacobian_result,
    validate_mmc_field,
)

__all__ = ["generate_jacobian"]

_DETECTOR_RADIUS_MM = 1.0
_DETECTOR_AREA_MM2 = np.pi * _DETECTOR_RADIUS_MM**2


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


def _validate_complete_detected_history(
    detected_photons: Mapping[str, ArrayLike],
    source_index: int,
) -> None:
    """Require all detected photons to be present for absolute detector measurements."""
    detected_counts = np.asarray(
        detected_photons["detected_counts"],
        dtype=np.int64,
    ).reshape(-1)
    saved_counts = np.asarray(
        detected_photons["saved_counts"],
        dtype=np.int64,
    ).reshape(-1)

    if detected_counts.shape != saved_counts.shape:
        raise ValueError("MMC history detected/saved count arrays must have matching shapes")

    if np.any(detected_counts < 0) or np.any(saved_counts < 0):
        raise ValueError("MMC history photon counts must be non-negative")

    if np.any(saved_counts > detected_counts):
        raise ValueError("MMC history reports more saved photons than detected photons")

    truncated = saved_counts < detected_counts
    if np.any(truncated):
        block = int(np.flatnonzero(truncated)[0])

        raise RuntimeError(
            "MMC detected-photon history was truncated for "
            f"source {source_index}, block {block}: "
            f"detected={detected_counts[block]}, "
            f"saved={saved_counts[block]}. "
            "Green_sd and mea0 require the complete detected-photon history; "
            "increase MMC maxdetphoton."
        )


def _compute_element_volumes(
    nodes: ArrayLike,
    elements: ArrayLike,
) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=np.intp)

    p0 = nodes[elements[:, 0]]
    p1 = nodes[elements[:, 1]]
    p2 = nodes[elements[:, 2]]
    p3 = nodes[elements[:, 3]]

    return (
        np.abs(
            np.einsum(
                "ij,ij->i",
                p1 - p0,
                np.cross(p2 - p0, p3 - p0),
            )
        )
        / 6.0
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

    element_volumes = _compute_element_volumes(node_array, element_array)

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


def _interpolate_nodal_field_in_tetrahedron(
    point: ArrayLike,
    element_index: int,
    nodes: ArrayLike,
    elements: ArrayLike,
    nodal_field: ArrayLike,
) -> float:
    """Barycentrically interpolate a nodal field at a point inside a tetrahedron."""
    point = np.asarray(point, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=np.intp)
    field = np.asarray(nodal_field, dtype=float)

    element_nodes = elements[element_index]
    tetra = nodes[element_nodes]

    # p = v0 + B @ [w1, w2, w3]
    v0 = tetra[0]
    B = np.column_stack(
        (
            tetra[1] - v0,
            tetra[2] - v0,
            tetra[3] - v0,
        )
    )

    w123 = np.linalg.solve(B, point - v0)
    bary = np.array(
        [
            1.0 - w123.sum(),
            w123[0],
            w123[1],
            w123[2],
        ],
        dtype=float,
    )

    # Registration claims this point belongs to element_index.
    # Allow tiny floating-point violations only.
    tol = 1e-5
    if np.any(bary < -tol) or np.any(bary > 1.0 + tol):
        raise ValueError(f"Point is not inside element {element_index}: barycentric coordinates={bary}")

    return float(np.dot(bary, field[element_nodes]))


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


def _calculate_detected_mean_partial_pathlengths(
    detected_photons: Mapping[str, ArrayLike],
    photon_weights: ArrayLike,
    detector_weight_sums: ArrayLike,
    detector_count: int,
) -> np.ndarray:
    """Detected-weighted mean partial path length per detector and medium."""
    weights = np.asarray(photon_weights, dtype=float)
    weight_sums = np.asarray(detector_weight_sums, dtype=float)
    detector_ids = np.asarray(detected_photons["detid"], dtype=np.intp)
    partial_paths = np.asarray(detected_photons["ppath"], dtype=float)
    unitinmm = float(detected_photons.get("unitinmm", 1.0))

    partial_paths_mm = partial_paths * unitinmm
    medium_count = partial_paths.shape[1]

    result = np.full(
        (detector_count, medium_count),
        np.nan,
        dtype=float,
    )

    for detector_index in range(detector_count):
        mask = detector_ids == detector_index + 1
        if weight_sums[detector_index] == 0:
            continue

        result[detector_index] = (weights[mask, None] * partial_paths_mm[mask]).sum(axis=0) / weight_sums[
            detector_index
        ]

    return result


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
    basis_order: int = 1,
    replay: bool = False,
    compute_backend: str | None = None,
    gpu_id: int | None = None,
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

    backend_args: list[str] = []
    if compute_backend is not None:
        if compute_backend not in {"cuda", "opencl", "sse"}:
            raise ValueError("compute_backend must be one of {'cuda', 'opencl', 'sse'} or None")
        backend_args.extend(["-c", compute_backend])
    if gpu_id is not None:
        backend_args.extend(["-G", str(gpu_id)])

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
    base_config["basisorder"] = basis_order
    base_config["seed"] = 123456789

    if replay:
        result = _generate_replay_jacobian(
            inputs=inputs,
            base_config=base_config,
            timeout=float(timeout),
            detector_radius_mm=_DETECTOR_RADIUS_MM,
            backend_args=backend_args,
        )
        if resolved_save_path is not None:
            save_jacobian_result(resolved_save_path, result)
        return result

    field_count = len(inputs.nodes) if basis_order == 1 else len(inputs.elements)
    print(inputs.selected_properties)
    print(base_config["prop"])
    detector_positions_with_radius = np.column_stack(
        (inputs.detector_positions, np.full(len(inputs.detector_positions), _DETECTOR_RADIUS_MM))
    )

    source_count = len(inputs.source_positions)
    detector_count = len(inputs.detector_positions)
    row_count = source_count * detector_count
    detected_photon_counts = np.zeros(row_count, dtype=np.int64)
    detected_photon_ess = np.zeros(row_count, dtype=float)

    node_volumes = _calculate_node_volumes(inputs.nodes, inputs.elements)

    green_source = np.zeros((source_count, field_count), dtype=float)
    green_detector = np.zeros((detector_count, field_count), dtype=float)
    green_source_detector = np.zeros((row_count, 1), dtype=float)
    # Diagnostic alternative denominator:
    # source volumetric fluence interpolated exactly at detector position.
    green_source_detector_fluence = np.zeros((row_count, 1), dtype=float)
    measurements_zero = np.zeros((row_count, 1), dtype=float)

    medium_count = inputs.selected_properties.shape[0] - 1
    replay_mean_partial_pathlengths = np.full(
        (row_count, medium_count),
        np.nan,
        dtype=float,
    )

    with TemporaryDirectory(prefix="mmcnirs-jacobian-") as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        selected_channel_rows = set(np.asarray(inputs.channel_indices, dtype=int).reshape(-1))
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

            source_fluence, detected_photons = read_cli_output(output_stub)
            _validate_complete_detected_history(detected_photons, source_index)
            source_fluence = validate_mmc_field(source_fluence, field_count, f"source {source_index}")
            green_source[source_index] = source_fluence
            for detector_index in range(detector_count):
                row = source_index * detector_count + detector_index

                green_source_detector_fluence[row, 0] = _interpolate_nodal_field_in_tetrahedron(
                    point=inputs.detector_positions[detector_index],
                    element_index=int(inputs.detector_elements[detector_index]),
                    nodes=inputs.nodes,
                    elements=inputs.elements,
                    nodal_field=source_fluence,
                )

            row_start = source_index * detector_count
            row_stop = row_start + detector_count

            photon_weights = compute_detected_photon_weights(
                detected_photons,
                optical_properties=inputs.selected_properties,
            )

            detector_weight_sums = _sum_detected_photon_weights(
                detected_photons,
                photon_weights,
                detector_count,
            )
            measurements_zero[row_start:row_stop, 0] = detector_weight_sums / inputs.photon_count

            detector_mean_partial_pathlengths = _calculate_detected_mean_partial_pathlengths(
                detected_photons,
                photon_weights,
                detector_weight_sums,
                detector_count,
            )
            replay_mean_partial_pathlengths[row_start:row_stop] = detector_mean_partial_pathlengths

            detector_ids = np.asarray(detected_photons["detid"], dtype=int)
            for detector_index in range(detector_count):
                row = source_index * detector_count + detector_index
                mask = detector_ids == detector_index + 1

                detected_photon_counts[row] = np.count_nonzero(mask)

                w = photon_weights[mask]
                if w.size and np.sum(w * w) > 0:
                    detected_photon_ess[row] = np.sum(w) ** 2 / np.sum(w * w)
            for detector_index in np.flatnonzero(detector_weight_sums == 0):
                row = source_index * detector_count + detector_index
                detected_count = np.sum(detector_ids == detector_index + 1)
                message = (
                    f"source={source_index}, detector={detector_index}, detected_count={detected_count}, weight_sum=0."
                )
                if row in selected_channel_rows:
                    raise RuntimeError("Selected " + message)
                warnings.warn("Unselected " + message, RuntimeWarning, stacklevel=2)

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
                # "srctype": "disk",
                # "srcparam1": [_DETECTOR_RADIUS_MM, 0.0, 0.0, 0.0]
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

            detector_fluence = read_flux(output_stub.with_suffix(".dat"))
            detector_fluence = validate_mmc_field(detector_fluence, field_count, f"detector {detector_index}")
            green_detector[detector_index] = detector_fluence

    # Current reflectance-normalized construction
    jacobian = _calculate_jacobian(green_source, green_detector, green_source_detector, node_volumes)
    # Diagnostic: use volumetric source fluence at detector location
    jacobian_fluence_denominator = _calculate_jacobian(
        green_source, green_detector, green_source_detector_fluence, node_volumes
    )

    jacobian_effective_pathlength = -jacobian.sum(axis=1, keepdims=True)
    jacobian_fluence_effective_pathlength = -jacobian_fluence_denominator.sum(axis=1, keepdims=True)

    replay_mean_pathlength = replay_mean_partial_pathlengths.sum(axis=1, keepdims=True)

    pathlength_ratio = np.divide(
        jacobian_effective_pathlength,
        replay_mean_pathlength,
        out=np.full_like(jacobian_effective_pathlength, np.nan),
        where=replay_mean_pathlength > 0,
    )
    pathlength_ratio_fluence_denominator = np.divide(
        jacobian_fluence_effective_pathlength,
        replay_mean_pathlength,
        out=np.full_like(jacobian_fluence_effective_pathlength, np.nan),
        where=replay_mean_pathlength > 0,
    )
    Green_sd_fluence_over_reflectance = np.divide(
        green_source_detector_fluence,
        green_source_detector,
        out=np.full_like(
            green_source_detector_fluence,
            np.nan,
        ),
        where=green_source_detector > 0,
    )
    channel_normalization = np.divide(
        replay_mean_pathlength,
        jacobian_effective_pathlength,
        out=np.full_like(replay_mean_pathlength, np.nan),
        where=jacobian_effective_pathlength > 0,
    )
    result = {
        "Green_d": green_detector,
        "Green_s": green_source,
        "Green_sd": green_source_detector,
        "Green_sd_fluence": green_source_detector_fluence,
        "J": jacobian,
        "J_fluence_denominator": jacobian_fluence_denominator,
        "replay_mean_pathlength": replay_mean_pathlength,
        "jacobian_effective_pathlength": jacobian_effective_pathlength,
        "jacobian_fluence_effective_pathlength": jacobian_fluence_effective_pathlength,
        "pathlength_ratio": pathlength_ratio,
        "pathlength_ratio_fluence_denominator": pathlength_ratio_fluence_denominator,
        "Green_sd_fluence_over_reflectance": Green_sd_fluence_over_reflectance,
        "channel_normalization": channel_normalization,
        "replay_mean_partial_pathlengths": replay_mean_partial_pathlengths,
        "channelidx": inputs.channel_indices,
        "mea0": measurements_zero,
        "sourcepos": inputs.source_positions,
        "detpos": detector_positions_with_radius,
        "detnorms": inputs.detector_directions,
        "sourcedir": inputs.source_directions,
        "detected_photon_counts": detected_photon_counts,
        "detected_photon_ess": detected_photon_ess,
    }
    if resolved_save_path is not None:
        save_jacobian_result(resolved_save_path, result)
    return result


def _generate_replay_jacobian(
    inputs,
    base_config: Mapping[str, Any],
    timeout: float,
    detector_radius_mm: float,
    backend_args: list[str] | None = None,
) -> dict[str, np.ndarray]:
    backend_args = [] if backend_args is None else list(backend_args)
    """Generate absorption Jacobian using MMC detected-photon replay."""
    node_volumes = _calculate_node_volumes(inputs.nodes, inputs.elements)
    source_count = len(inputs.source_positions)
    detector_count = len(inputs.detector_positions)
    node_count = len(inputs.nodes)
    row_count = source_count * detector_count

    detector_positions_with_radius = np.column_stack(
        (
            inputs.detector_positions,
            np.full(detector_count, detector_radius_mm),
        )
    )

    # Full source-major matrix.
    # Unselected source-detector rows remain zero.
    jacobian = np.zeros((row_count, node_count), dtype=float)
    measurements_zero = np.zeros((row_count, 1), dtype=float)
    detected_photon_counts = np.zeros(row_count, dtype=np.int64)
    detected_photon_ess = np.zeros(row_count, dtype=float)
    medium_count = inputs.selected_properties.shape[0] - 1
    replay_mean_partial_pathlengths = np.full((row_count, medium_count), np.nan, dtype=float)
    selected_rows = np.unique(np.asarray(inputs.channel_indices, dtype=int).reshape(-1))
    with TemporaryDirectory(prefix="mmcnirs-replay-jacobian-") as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)

        for source_index in tqdm(
            range(source_count),
            desc="MMC replay sources",
            unit="source",
        ):
            # Which selected channels belong to this source?
            source_row_start = source_index * detector_count
            source_row_stop = source_row_start + detector_count

            source_selected_rows = [row for row in selected_rows if source_row_start <= row < source_row_stop]

            # If this source has no selected channels, skip it.
            if not source_selected_rows:
                continue

            source_stub = temporary_directory / f"source_{source_index:04d}"

            source_config = base_config | {
                "srcpos": inputs.source_positions[source_index].tolist(),
                "e0": int(inputs.source_elements[source_index]) + 1,
                "srcdir": inputs.source_directions[source_index].tolist(),
                "detpos": detector_positions_with_radius.tolist(),
                # CRITICAL for replay
                "issaveseed": 1,
                "issavedet": 1,
                "issaveexit": 1,
                # forward output itself is not important here,
                # but fluence is fine
                "outputtype": "fluence",
            }

            source_config_path = source_stub.with_suffix(".json")

            mmc_to_json(
                source_config,
                source_config_path,
            )

            completed = run_mmc(
                source_config_path, working_directory=temporary_directory, timeout=timeout, extra_args=backend_args
            )
            if source_index == 0:
                print("MMC FORWARD STDOUT:")
                print(completed.stdout)
                print("MMC FORWARD STDERR:")
                print(completed.stderr)

            # Read forward photon history.
            _, detected_photons = read_cli_output(source_stub)

            _validate_complete_detected_history(
                detected_photons,
                source_index,
            )

            photon_weights = compute_detected_photon_weights(
                detected_photons,
                optical_properties=(inputs.selected_properties),
            )

            detector_weight_sums = _sum_detected_photon_weights(
                detected_photons,
                photon_weights,
                detector_count,
            )

            # baseline intensity
            measurements_zero[source_row_start:source_row_stop, 0] = detector_weight_sums / inputs.photon_count

            # existing replay path-length diagnostic
            detector_mean_partial_pathlengths = _calculate_detected_mean_partial_pathlengths(
                detected_photons,
                photon_weights,
                detector_weight_sums,
                detector_count,
            )

            replay_mean_partial_pathlengths[source_row_start:source_row_stop] = detector_mean_partial_pathlengths

            detector_ids = np.asarray(
                detected_photons["detid"],
                dtype=int,
            )

            for detector_index in range(detector_count):
                row = source_index * detector_count + detector_index
                mask = detector_ids == detector_index + 1
                detected_photon_counts[row] = np.count_nonzero(mask)
                w = photon_weights[mask]
                if w.size and np.sum(w * w) > 0:
                    detected_photon_ess[row] = np.sum(w) ** 2 / np.sum(w * w)

            # The forward source run should have created:
            source_history_path = source_stub.with_suffix(".mch")

            if not source_history_path.is_file():
                raise RuntimeError(
                    f"MMC forward replay run did not create {source_history_path.name}. Check issaveseed/DoSaveSeed."
                )

            # Replay ONLY selected detectors for this source.
            for row in source_selected_rows:
                detector_index = row - source_row_start

                if detector_weight_sums[detector_index] <= 0:
                    raise RuntimeError(
                        "Selected replay channel has zero detected weight: "
                        f"source={source_index}, detector={detector_index}"
                    )

                replay_stub = temporary_directory / (f"replay_s{source_index:04d}_d{detector_index:04d}")

                # Same physical forward configuration.
                replay_config = dict(source_config)
                replay_config_path = replay_stub.with_suffix(".json")
                mmc_to_json(replay_config, replay_config_path)

                completed = run_mmc(
                    replay_config_path,
                    working_directory=temporary_directory,
                    timeout=timeout,
                    extra_args=[
                        *backend_args,
                        "-E",
                        source_history_path.name,
                        "-P",
                        str(detector_index + 1),
                        "-O",
                        "L",
                    ],
                )
                if source_index == 0:
                    print("MMC REPLAY STDOUT:")
                    print(completed.stdout)
                    print("MMC REPLAY STDERR:")
                    print(completed.stderr)

                replay_field = read_flux(replay_stub.with_suffix(".dat"))

                replay_field = validate_mmc_field(
                    replay_field,
                    node_count,
                    (f"replay source {source_index}, detector {detector_index}"),
                )

                # Official mmcjmua.m:
                #   outputtype = 'wl'
                #   Ja = -jacob.data
                jacobian[row] = -replay_field

    replay_mean_pathlength = replay_mean_partial_pathlengths.sum(axis=1, keepdims=True)
    jacobian_effective_pathlength = -jacobian.sum(axis=1, keepdims=True)
    pathlength_ratio = np.divide(
        jacobian_effective_pathlength,
        replay_mean_pathlength,
        out=np.full_like(replay_mean_pathlength, np.nan),
        where=replay_mean_pathlength > 0,
    )

    return {
        "J": jacobian,
        "node_volumes": node_volumes,
        "replay_mean_pathlength": replay_mean_pathlength,
        "jacobian_effective_pathlength": jacobian_effective_pathlength,
        "pathlength_ratio": pathlength_ratio,
        "replay_mean_partial_pathlengths": replay_mean_partial_pathlengths,
        "channelidx": inputs.channel_indices,
        "mea0": measurements_zero,
        "sourcepos": inputs.source_positions,
        "detpos": detector_positions_with_radius,
        "detnorms": inputs.detector_directions,
        "sourcedir": inputs.source_directions,
        "detected_photon_counts": detected_photon_counts,
        "detected_photon_ess": detected_photon_ess,
    }
