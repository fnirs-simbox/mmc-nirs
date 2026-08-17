"""Prepare fNIRS probes for Jacobian generation."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mmc_nirs.utils.prepared_input_io import (
    load_prepared_input,
    resolve_prepared_input_path,
    save_prepared_input,
)

from .probe_utils import (
    _pairing_indices,
    _plot_probe_registration,
    load_channel_pairs_from_snirf as load_channel_pairs_from_snirf,
)
from .register_probe import register_probe

_PROBE_ARCHIVE_KEYS = {
    "sourcepos",
    "detpos",
    "sourcedir",
    "detnorms",
    "source_elements",
    "detector_elements",
    "channel_pairings",
    "short_separation_indices",
    "long_separation_indices",
}


def prepare_jacobian_probe(
    source_positions: ArrayLike,
    detector_positions: ArrayLike,
    prepared_mesh: Mapping[str, ArrayLike],
    units: str,
    orientation: str,
    channel_pairings: ArrayLike,
    short_separation_flag: str,
    short_separation_arg: float | list[int],
    experiment_config: Mapping[str, Any],
    embedding_step: float = 0.1,
    max_embedding_steps: int = 1_000,
    plot: bool = False,
    save_probe: bool = True,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """Register a probe to a prepared mesh for Jacobian generation.

    Existing prepared probe archives are reused unless ``overwrite`` is true.
    Otherwise, probe coordinates are registered to the prepared mesh and
    short- and long-separation channels are identified from registered
    coordinates.
    """
    archive_path = resolve_prepared_input_path(experiment_config, "probefile")
    if archive_path.is_file() and not overwrite:
        return load_prepared_input(archive_path, _PROBE_ARCHIVE_KEYS)

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

    try:
        mesh_nodes = prepared_mesh["nodes"]
        mesh_elements = prepared_mesh["elements"]
    except (KeyError, TypeError) as error:
        raise ValueError("prepared_mesh must contain 'nodes' and 'elements'") from error

    pairs = np.asarray(channel_pairings)
    if pairs.ndim != 2 or pairs.shape[0] == 0 or pairs.shape[1] != 2:
        raise ValueError("channel_pairings must be a non-empty array with shape (n_channels, 2)")
    if not np.issubdtype(pairs.dtype, np.integer):
        if not np.all(np.isfinite(pairs)) or not np.all(pairs == np.floor(pairs)):
            raise ValueError("channel_pairings must contain integer indices")
    pairs = pairs.astype(np.intp, copy=True)
    if normalized_flag == "index":
        short_indices = np.asarray(short_separation_arg, dtype=np.intp)
        if np.any(short_indices < 0) or np.any(short_indices >= len(pairs)):
            raise ValueError("short-separation channel indices are out of range")

    (
        registered_sources,
        registered_detectors,
        source_directions,
        detector_directions,
        source_elements,
        detector_elements,
    ) = register_probe(
        source_positions,
        detector_positions,
        mesh_nodes,
        mesh_elements,
        probe_orientation=orientation,
        probe_units=units,
        embedding_step=embedding_step,
        max_embedding_steps=max_embedding_steps,
    )

    source_indices = _pairing_indices(pairs[:, 0], len(registered_sources), "source")
    detector_indices = _pairing_indices(pairs[:, 1], len(registered_detectors), "detector")

    if normalized_flag == "distance":
        distances = np.linalg.norm(
            registered_sources[source_indices] - registered_detectors[detector_indices],
            axis=1,
        )
        short_indices = np.flatnonzero(distances <= short_separation_arg)

    long_indices = np.setdiff1d(np.arange(len(pairs), dtype=np.intp), short_indices)
    probe = {
        "sourcepos": registered_sources,
        "detpos": registered_detectors,
        "sourcedir": source_directions,
        "detnorms": detector_directions,
        "source_elements": source_elements,
        "detector_elements": detector_elements,
        "channel_pairings": pairs,
        "short_separation_indices": short_indices,
        "long_separation_indices": long_indices,
    }
    if plot:
        _plot_probe_registration(
            mesh_nodes,
            mesh_elements,
            registered_sources,
            registered_detectors,
            source_directions,
            detector_directions,
            source_indices,
            detector_indices,
        )
    if save_probe:
        save_prepared_input(archive_path, probe)
    return probe
