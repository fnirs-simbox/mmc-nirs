"""Prepare fNIRS probes for Jacobian generation."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mmc_nirs.utils.mesh_utils import validate_prepared_mesh
from mmc_nirs.utils.prepared_input_io import (
    load_npz_archive,
    resolve_prepared_input_path,
    save_npz_archive,
)
from mmc_nirs.utils.probe_utils import (
    PREPARED_PROBE_KEYS,
    _plot_probe_registration,
    as_channel_pairing_array,
    load_channel_pairs_from_snirf as load_channel_pairs_from_snirf,
    normalize_channel_pairings,
    validate_probe_settings,
)
from .register_probe import register_probe

_PROBE_ARCHIVE_KEYS = PREPARED_PROBE_KEYS | {
    "short_separation_indices",
    "long_separation_indices",
}


def prepare_jacobian_probe(
    source_positions: ArrayLike,
    detector_positions: ArrayLike,
    prepared_mesh: Mapping[str, ArrayLike],
    channel_pairings: ArrayLike,
    experiment_config: Mapping[str, Any],
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
        Prepared tetrahedral mesh containing ``"nodes"``, ``"elements"``,
        ``"element_tissue_ids"``, ``"ordered_tissue_ids"``, and
        ``"ordered_tissues"``. Elements must have shape ``(n_elements, 4)``
        with zero-based indices. The complete mesh is validated even when a
        previously prepared probe archive is reused.
    channel_pairings : array-like
        Source-detector index pairs with shape ``(n_channels, 2)``. Indices may
        be zero-based or one-based.
    experiment_config : mapping
        Experiment configuration containing:

        - ``experiment_dir``: path beneath which prepared outputs are stored.
        - ``filepaths.probefile``: path of the prepared probe NPZ archive,
          relative to ``experiment_dir`` unless absolute.
        - ``probe_settings.probe_units``: units used by ``source_positions``
          and ``detector_positions``; one of ``"mm"``, ``"cm"``, or ``"m"``.
          Prepared mesh coordinates are assumed to be millimetres.
        - ``probe_settings.probe_orientation``: three-letter anatomical
          orientation code describing the probe axes, such as ``"RAS"`` or
          ``"LIA"``.
        - ``probe_settings.short_separation_flag``: ``"distance"`` or
          ``"index"``, selecting how short-separation channels are identified.
        - ``probe_settings.short_separation_arg``: a finite, non-negative float
          distance in millimetres when the flag is ``"distance"``; when the
          flag is ``"index"``, a list of zero-based channel indices.
        - ``probe_settings.embedding_step``: positive scalar embedding distance
          in millimetres applied per iteration.
        - ``probe_settings.max_embedding_steps``: non-negative integer maximum
          number of embedding iterations.

        ``probe_settings`` and all six of its fields are required even when a
        previously prepared probe archive is reused.
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
    probe_settings = validate_probe_settings(experiment_config)
    normalized_flag = probe_settings["short_separation_flag"]
    short_separation_arg = probe_settings["short_separation_arg"]
    mesh = validate_prepared_mesh(prepared_mesh)

    archive_path = resolve_prepared_input_path(experiment_config, "probefile")
    if archive_path.is_file() and not overwrite:
        return load_npz_archive(archive_path, _PROBE_ARCHIVE_KEYS)

    pairs = as_channel_pairing_array(channel_pairings)
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
        mesh["nodes"],
        mesh["elements"],
        probe_orientation=probe_settings["probe_orientation"],
        probe_units=probe_settings["probe_units"],
        embedding_step=probe_settings["embedding_step"],
        max_embedding_steps=probe_settings["max_embedding_steps"],
    )

    normalized_pairings = normalize_channel_pairings(
        pairs,
        len(registered_sources),
        len(registered_detectors),
    )
    source_indices = normalized_pairings[:, 0]
    detector_indices = normalized_pairings[:, 1]

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
        "channel_pairings": normalized_pairings,
        "short_separation_indices": short_indices,
        "long_separation_indices": long_indices,
    }
    if plot:
        _plot_probe_registration(
            mesh["nodes"],
            mesh["elements"],
            registered_sources,
            registered_detectors,
            source_directions,
            detector_directions,
            source_indices,
            detector_indices,
        )
    if save_probe:
        save_npz_archive(archive_path, probe)
    return probe
