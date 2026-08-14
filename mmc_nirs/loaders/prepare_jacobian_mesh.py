"""Prepare tetrahedral head meshes for Jacobian generation."""

import numpy as np
from numpy.typing import ArrayLike

from mmc_nirs.utils.mesh_utils import _as_coordinate_array, _as_element_array, make_orientation_matrices


def prepare_jacobian_mesh(
    nodes: ArrayLike,
    elements: ArrayLike,
    node_tissue_values: ArrayLike,
    orientation: str,
    units: str,
    scalp_idx: int = 5,
    skull_idx: int = 4,
    CSF_idx: int = 3,
    gray_matter_idx: int = 2,
    white_matter_idx: int = 1,
) -> dict[str, np.ndarray]:
    """Normalize a tetrahedral head mesh for Jacobian generation.

    Coordinates are converted to millimetres and reoriented to RAS. Element
    indices are saved zero-based, and the supplied tissue labels are remapped
    to white matter=1, gray matter=2, CSF=3, skull=4, and scalp=5.

    Parameters
    ----------
    nodes : array-like, shape (n_nodes, 3)
        Mesh-node coordinates.
    elements : array-like, shape (n_elements, 4)
        Zero- or one-based tetrahedral node indices.
    node_tissue_values : array-like, shape (n_nodes,)
        Tissue label for each node.
    orientation : str
        Three-letter anatomical orientation code, such as ``"RAS"`` or
        ``"LIA"``.
    units : {"mm", "cm", "m"}
        Units of the input node coordinates.
    """
    node_array = _as_coordinate_array(nodes, "nodes")
    element_array = _as_element_array(elements, len(node_array), "elements", allow_extra_columns=False)

    tissue_array = np.asarray(node_tissue_values)
    if tissue_array.ndim != 1 or tissue_array.shape[0] != len(node_array):
        raise ValueError("node_tissue_values must be a one-dimensional value for every node")

    input_labels = (white_matter_idx, gray_matter_idx, CSF_idx, skull_idx, scalp_idx)
    if len(set(input_labels)) != len(input_labels):
        raise ValueError("tissue indices must be unique")
    unknown_labels = np.setdiff1d(np.unique(tissue_array), input_labels)
    if unknown_labels.size:
        raise ValueError(f"node_tissue_values contains unknown tissue labels: {unknown_labels.tolist()}")

    normalized_tissues = np.empty(tissue_array.shape, dtype=np.uint8)
    for normalized_label, input_label in enumerate(input_labels, start=1):
        normalized_tissues[tissue_array == input_label] = normalized_label

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

    return {"nodes": ras_nodes, "elements": element_array, "node_tissue_values": normalized_tissues}
