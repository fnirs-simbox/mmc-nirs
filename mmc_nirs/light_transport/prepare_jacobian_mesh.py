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
    make_orientation_matrices,
    ordered_tissue_arrays,
)
from mmc_nirs.utils.prepared_input_io import (
    load_npz_archive,
    resolve_prepared_input_path,
    save_npz_archive,
)


def prepare_jacobian_mesh(
    nodes: ArrayLike,
    elements: ArrayLike,
    element_tissue_ids: ArrayLike,
    orientation: str,
    units: str,
    experiment_config: Mapping[str, Any],
    save_mesh: bool = True,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """Normalize a tetrahedral head mesh for Jacobian generation.

    Coordinates are converted to millimetres and reoriented to RAS. Element
    indices are saved zero-based. Element tissue IDs are positional MMC medium
    IDs declared by ``experiment_config["ordered_tissues"]``.

    Parameters
    ----------
    nodes : array-like, shape (n_nodes, 3)
        Mesh-node coordinates.
    elements : array-like, shape (n_elements, 4)
        Zero- or one-based tetrahedral node indices.
    element_tissue_ids : array-like, shape (n_elements,)
        Positional MMC medium ID for each tetrahedral element.
    orientation : str
        Three-letter anatomical orientation code, such as ``"RAS"`` or
        ``"LIA"``.
    units : {"mm", "cm", "m"}
        Units of the input node coordinates.
    experiment_config : mapping
        Experiment configuration with top-level ``experiment_dir`` and
        ``ordered_tissues`` entries and a ``filepaths`` mapping containing
        ``meshfile``. Tissue IDs must be contiguous from zero.
    save_mesh : bool, default=True
        Whether to save the prepared mesh to its configured path.
    overwrite : bool, default=False
        Whether to recompute when a prepared mesh archive already exists.
    """
    archive_path = resolve_prepared_input_path(experiment_config, "meshfile")
    if archive_path.is_file() and not overwrite:
        return load_npz_archive(archive_path, PREPARED_MESH_KEYS)

    node_array = as_coordinate_array(nodes, "nodes")
    element_array = as_element_array(elements, len(node_array), "elements", allow_extra_columns=False)

    tissue_id_array = as_element_tissue_id_array(element_tissue_ids, len(element_array))
    try:
        ordered_tissue_ids, ordered_tissues = ordered_tissue_arrays(experiment_config["ordered_tissues"])
    except KeyError as error:
        raise ValueError("experiment_config is missing required field: ordered_tissues") from error
    unknown_ids = np.setdiff1d(np.unique(tissue_id_array), ordered_tissue_ids)
    if unknown_ids.size:
        raise ValueError(f"element_tissue_ids contains IDs not represented by ordered_tissues: {unknown_ids.tolist()}")

    normalized_units = units.lower()
    if normalized_units not in {"mm", "m", "cm"}:
        raise ValueError("units must be either 'mm', 'cm' or 'm'")
    if normalized_units == "m":
        unit_scale = 1_000.0
    elif normalized_units == "cm":
        unit_scale = 10.0
    else:
        unit_scale = 1.0

    orientation_code = orientation.upper()
    try:
        orientation_matrix = make_orientation_matrices()[orientation_code]
    except KeyError as error:
        raise ValueError(f"Unknown mesh orientation {orientation!r}") from error
    ras_nodes = node_array * unit_scale @ orientation_matrix.T

    prepared = {
        "nodes": ras_nodes,
        "elements": element_array,
        "element_tissue_ids": tissue_id_array,
        "ordered_tissue_ids": ordered_tissue_ids,
        "ordered_tissues": ordered_tissues,
    }
    if save_mesh:
        save_npz_archive(archive_path, prepared)
    return prepared
