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

    Parameters
    ----------
    source_positions : array-like
        Source coordinates with shape ``(n_sources, 3)``.
    detector_positions : array-like
        Detector coordinates with shape ``(n_detectors, 3)``.
    prepared_mesh : mapping
        Prepared tetrahedral mesh containing a ``"nodes"`` array with shape
        ``(n_nodes, 3)`` and an ``"elements"`` array with shape
        ``(n_elements, 4)``. Element indices may be zero-based or one-based.
    units : {"mm", "cm", "m"}
        Unit used by ``source_positions`` and ``detector_positions``. This is a
        scalar string; prepared mesh coordinates are assumed to be millimetres.
    orientation : str
        Three-letter anatomical orientation code describing the probe axes,
        such as ``"RAS"`` or ``"LIA"``. This is a scalar string containing one
        letter from each anatomical axis pair.
    channel_pairings : array-like
        Source-detector index pairs with shape ``(n_channels, 2)``. Indices may
        be zero-based or one-based.
    short_separation_flag : {"distance", "index"}
        Scalar string selecting how short-separation channels are identified.
    short_separation_arg : float or list[int]
        A finite, non-negative scalar distance in millimetres when
        ``short_separation_flag`` is ``"distance"``. When the flag is
        ``"index"``, this must be a one-dimensional list of zero-based channel
        indices with length ``n_short_channels``.
    experiment_config : mapping
        Configuration containing a ``"filepaths"`` mapping. That mapping must
        contain scalar path values for ``"experiment_dir"`` and ``"probefile"``;
        ``"probefile"`` must name an NPZ file.
    embedding_step : float, default=0.1
        Positive scalar embedding distance in millimetres applied per iteration.
    max_embedding_steps : int, default=1000
        Non-negative scalar maximum number of embedding iterations.
    plot : bool, default=False
        Scalar flag indicating whether to create the registration diagnostic.
    save_probe : bool, default=True
        Scalar flag indicating whether to save the prepared probe NPZ archive.
    overwrite : bool, default=False
        Scalar flag indicating whether an existing prepared archive should be
        replaced instead of loaded.

    Returns
    -------
    dict[str, numpy.ndarray]
        Prepared probe arrays. Coordinate and direction fields have shapes
        ``(n_sources, 3)`` or ``(n_detectors, 3)``; element-index fields have
        shapes ``(n_sources,)`` or ``(n_detectors,)``; ``channel_pairings`` has
        shape ``(n_channels, 2)``; and separation-index fields are one-dimensional.
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
