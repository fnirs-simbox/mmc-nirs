"""Helpers for creating normalized MMC input files."""

from os import PathLike
from pathlib import Path

import numpy as np
import h5py
from numpy.typing import ArrayLike

from mmc_nirs.registration import make_orientation_matrices


def create_mesh_input(
    filename: str | PathLike[str],
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
    save_mesh: bool = False,
):
    """Validate and save a tetrahedral head mesh in the package format.

    Coordinates are converted to millimetres and reoriented to RAS. Element
    indices are saved zero-based, and the supplied tissue labels are remapped
    to white matter=1, gray matter=2, CSF=3, skull=4, and scalp=5.

    Parameters
    ----------
    filename : path-like
        Destination ``.npz`` file.
    nodes : array-like, shape (n_nodes, 3)
        Mesh-node coordinates.
    elements : array-like, shape (n_elements, 4)
        Zero- or one-based tetrahedral node indices.
    node_tissue_values : array-like, shape (n_nodes,)
        Tissue label for each node.
    orientation : str
        Three-letter anatomical orientation code, such as ``"RAS"`` or
        ``"LIA"``.
    units : {"mm", "m"}
        Units of the input node coordinates.
    """
    output_path = Path(filename)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("filename must have a .npz suffix")

    node_array = np.asarray(nodes, dtype=float)
    if node_array.ndim != 2 or node_array.shape[0] == 0 or node_array.shape[1] != 3:
        raise ValueError("nodes must be a non-empty array with shape (n_nodes, 3)")
    if not np.all(np.isfinite(node_array)):
        raise ValueError("nodes must contain only finite values")

    element_array = np.asarray(elements)
    if element_array.ndim != 2 or element_array.shape[0] == 0 or element_array.shape[1] != 4:
        raise ValueError("elements must be a non-empty array with shape (n_elements, 4)")
    if not np.issubdtype(element_array.dtype, np.integer):
        if not np.all(np.isfinite(element_array)) or not np.all(element_array == np.floor(element_array)):
            raise ValueError("elements must contain integer node indices")
    element_array = element_array.astype(np.intp, copy=True)
    if element_array.min() >= 1 and element_array.max() <= len(node_array):
        element_array -= 1
    if element_array.min() < 0 or element_array.max() >= len(node_array):
        raise ValueError("elements contains an out-of-range node index")

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

    if save_mesh:
        np.savez(
            output_path,
            nodes=ras_nodes,
            elements=element_array,
            node_tissue_values=normalized_tissues,
        )
    else:
        return {"nodes": ras_nodes, "elements": element_array, "node_tissue_values": normalized_tissues}


def create_probe_input(
    source_positions: ArrayLike,
    detector_positions: ArrayLike,
    units: str,
    orientation: str,
    channel_pairings: ArrayLike,
    short_separation_flag: str,
    short_separation_arg: float | list[int],
    save_probe: bool = False,
    filename: str | PathLike[str] | None = None,
) -> dict[str, np.ndarray | str] | None:
    """Normalize probe coordinates and identify short-separation channels.

    Coordinates are converted to millimetres and RAS orientation. Channel
    pairings may use either zero- or one-based source and detector indices.
    When ``short_separation_flag`` is ``"distance"``, channels whose
    source-detector distance is at most ``short_separation_arg`` are selected.
    When it is ``"index"``, ``short_separation_arg`` supplies those indices.
    """
    sources = _coordinate_array(source_positions, "source_positions")
    detectors = _coordinate_array(detector_positions, "detector_positions")

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
    if save_probe:
        if filename is None:
            raise ValueError("filename is required when save_probe is True")
        output_path = Path(filename)
        if output_path.suffix.lower() != ".npz":
            raise ValueError("filename must have a .npz suffix")
        np.savez(output_path, **probe)
        return None
    return probe


def _coordinate_array(values: ArrayLike, name: str) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] == 0 or coordinates.shape[1] != 3:
        raise ValueError(f"{name} must be a non-empty array with shape (n, 3)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} must contain only finite values")
    return coordinates.copy()


def _pairing_indices(indices: np.ndarray, size: int, coordinate_type: str) -> np.ndarray:
    if indices.min() >= 1 and indices.max() <= size:
        return indices - 1
    if indices.min() < 0 or indices.max() >= size:
        raise ValueError(f"channel_pairings contains an out-of-range {coordinate_type} index")
    return indices


def load_channel_pairs_from_snirf(snirf_file):
    with h5py.File(snirf_file, "r") as f:
        data_group = f["nirs"]["data1"]

        # Find measurementList groups and sort numerically
        ml_keys = [key for key in data_group.keys() if key.startswith("measurementList")]

        ml_keys = sorted(ml_keys, key=lambda x: int(x.replace("measurementList", "")))

        pairs = []

        for key in ml_keys:
            ml = data_group[key]

            source_idx = int(ml["sourceIndex"][()])
            detector_idx = int(ml["detectorIndex"][()])

            pairs.append([source_idx, detector_idx])

    return np.array(pairs, dtype=int)
