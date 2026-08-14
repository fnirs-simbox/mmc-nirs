"""Prepare fNIRS probes for Jacobian generation."""

from os import PathLike

import h5py
import numpy as np
from numpy.typing import ArrayLike

from mmc_nirs.utils.mesh_utils import _as_coordinate_array, make_orientation_matrices


def prepare_jacobian_probe(
    source_positions: ArrayLike,
    detector_positions: ArrayLike,
    units: str,
    orientation: str,
    channel_pairings: ArrayLike,
    short_separation_flag: str,
    short_separation_arg: float | list[int],
) -> dict[str, np.ndarray | str]:
    """Normalize probe coordinates and identify short-separation channels.

    Coordinates are converted to millimetres and RAS orientation. Channel
    pairings may use either zero- or one-based source and detector indices.
    When ``short_separation_flag`` is ``"distance"``, channels whose
    source-detector distance is at most ``short_separation_arg`` are selected.
    When it is ``"index"``, ``short_separation_arg`` supplies those indices.
    """
    sources = _as_coordinate_array(source_positions, "source_positions")
    detectors = _as_coordinate_array(detector_positions, "detector_positions")

    normalized_flag = short_separation_flag.lower()
    if normalized_flag not in {"distance", "index"}:
        raise ValueError("short_separation_flag must be either 'distance' or 'index'")
    if normalized_flag == "distance":
        if not isinstance(short_separation_arg, (float, np.floating)) or not np.isfinite(short_separation_arg):
            raise TypeError("short_separation_arg must be a finite float when short_separation_flag is 'distance'")
        if short_separation_arg < 0:
            raise ValueError("short_separation_arg distance must be non-negative")
    elif not isinstance(short_separation_arg, list) or not all(
        isinstance(index, (int, np.integer)) and not isinstance(index, bool) for index in short_separation_arg
    ):
        raise TypeError("short_separation_arg must be a list of integers when short_separation_flag is 'index'")

    unit_scales = {"mm": 1.0, "cm": 10.0, "m": 1_000.0}
    try:
        unit_scale = unit_scales[units.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("units must be either 'mm', 'cm', or 'm'") from error

    try:
        orientation_matrix = make_orientation_matrices()[orientation.upper()]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"Unknown probe orientation {orientation!r}") from error

    pairs = np.asarray(channel_pairings)
    if pairs.ndim != 2 or pairs.shape[0] == 0 or pairs.shape[1] != 2:
        raise ValueError("channel_pairings must be a non-empty array with shape (n_channels, 2)")
    if not np.issubdtype(pairs.dtype, np.integer):
        if not np.all(np.isfinite(pairs)) or not np.all(pairs == np.floor(pairs)):
            raise ValueError("channel_pairings must contain integer indices")
    pairs = pairs.astype(np.intp, copy=True)

    source_indices = _pairing_indices(pairs[:, 0], len(sources), "source")
    detector_indices = _pairing_indices(pairs[:, 1], len(detectors), "detector")
    sources_ras = sources * unit_scale @ orientation_matrix.T
    detectors_ras = detectors * unit_scale @ orientation_matrix.T

    if normalized_flag == "distance":
        distances = np.linalg.norm(sources_ras[source_indices] - detectors_ras[detector_indices], axis=1)
        short_indices = np.flatnonzero(distances <= short_separation_arg)
    else:
        short_indices = np.asarray(short_separation_arg, dtype=np.intp)
        if np.any(short_indices < 0) or np.any(short_indices >= len(pairs)):
            raise ValueError("short-separation channel indices are out of range")

    probe: dict[str, np.ndarray | str] = {
        "source_positions": sources_ras,
        "detector_positions": detectors_ras,
        "orientation": "RAS",
        "channel_pairings": pairs,
        "short_separation_indices": short_indices,
    }
    return probe


def _pairing_indices(indices: np.ndarray, size: int, coordinate_type: str) -> np.ndarray:
    if indices.min() >= 1 and indices.max() <= size:
        return indices - 1
    if indices.min() < 0 or indices.max() >= size:
        raise ValueError(f"channel_pairings contains an out-of-range {coordinate_type} index")
    return indices


def load_channel_pairs_from_snirf(snirf_file: str | PathLike[str]) -> np.ndarray:
    """Load source-detector channel pairings from a SNIRF file."""
    with h5py.File(snirf_file, "r") as snirf:
        data_group = snirf["nirs"]["data1"]
        measurement_keys = sorted(
            (key for key in data_group if key.startswith("measurementList")),
            key=lambda key: int(key.removeprefix("measurementList")),
        )
        pairs = [
            [int(data_group[key]["sourceIndex"][()]), int(data_group[key]["detectorIndex"][()])]
            for key in measurement_keys
        ]
    return np.asarray(pairs, dtype=int)
