"""Prepare tetrahedral head meshes for Jacobian generation."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mmc_nirs.utils.mesh_utils import (
    PREPARED_MESH_KEYS,
    as_coordinate_array,
    as_element_array,
    as_element_tissue_id_array,
    validate_mesh_settings,
)
from mmc_nirs.utils.prepared_input_io import (
    load_npz_archive,
    resolve_prepared_input_path,
    save_npz_archive,
)


def prepare_mesh(
    nodes: ArrayLike,
    elements: ArrayLike,
    element_tissue_ids: ArrayLike,
    experiment_config: Mapping[str, Any],
    save_mesh: bool = True,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """Normalize a tetrahedral head mesh for Jacobian generation.

    Coordinates are converted to millimetres and reoriented to RAS. Element
    indices are saved zero-based. Element tissue IDs are positional MMC medium
    IDs declared by ``experiment_config["mesh_settings"]["ordered_tissues"]``.

    Parameters
    ----------
    nodes : array-like, shape (n_nodes, 3)
        Mesh-node coordinates.
    elements : array-like, shape (n_elements, 4)
        Zero- or one-based tetrahedral node indices.
    element_tissue_ids : array-like, shape (n_elements,)
        Positional MMC medium ID for each tetrahedral element.
    experiment_config : mapping
        Experiment configuration containing:

        - ``experiment_dir``: path beneath which prepared outputs are stored.
        - ``filepaths.meshfile``: path of the prepared mesh NPZ archive,
          relative to ``experiment_dir`` unless absolute.
        - ``mesh_settings.mesh_orientation``: three-letter anatomical
          orientation code for ``nodes``, such as ``"RAS"`` or ``"LIA"``.
        - ``mesh_settings.mesh_units``: units of ``nodes``; one of ``"mm"``,
          ``"cm"``, or ``"m"``.
        - ``mesh_settings.ordered_tissues``: mapping from positional MMC medium
          IDs to tissue names. IDs must be contiguous from zero.

        ``mesh_settings`` and all three of its fields are required even when a
        previously prepared mesh archive is reused.
    save_mesh : bool, default=True
        Whether to save the prepared mesh to its configured path.
    overwrite : bool, default=False
        Whether to recompute when a prepared mesh archive already exists.
    """
    mesh_settings = validate_mesh_settings(experiment_config)

    archive_path = resolve_prepared_input_path(experiment_config, "meshfile")
    if archive_path.is_file() and not overwrite:
        return load_npz_archive(archive_path, PREPARED_MESH_KEYS)

    node_array = as_coordinate_array(nodes, "nodes")
    element_array = as_element_array(elements, len(node_array), "elements", allow_extra_columns=False)

    tissue_id_array = as_element_tissue_id_array(element_tissue_ids, len(element_array))
    unknown_ids = np.setdiff1d(np.unique(tissue_id_array), mesh_settings["ordered_tissue_ids"])
    if unknown_ids.size:
        raise ValueError(f"element_tissue_ids contains IDs not represented by ordered_tissues: {unknown_ids.tolist()}")

    ras_nodes = node_array * mesh_settings["unit_scale"] @ mesh_settings["orientation_matrix"].T

    prepared = {
        "nodes": ras_nodes,
        "elements": element_array,
        "element_tissue_ids": tissue_id_array,
        "ordered_tissue_ids": mesh_settings["ordered_tissue_ids"],
        "ordered_tissues": mesh_settings["ordered_tissues"],
    }
    if save_mesh:
        save_npz_archive(archive_path, prepared)
    return prepared
