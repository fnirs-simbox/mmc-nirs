"""Probe-to-mesh registration utilities."""

import numpy as np
import trimesh
from numpy.typing import ArrayLike

from mmc_nirs.utils.mesh_utils import (
    _as_coordinate_array,
    _as_element_array,
    _find_containing_elements,
    make_orientation_matrices,
)


def register_probe(
    source_coordinates: ArrayLike,
    detector_coordinates: ArrayLike,
    mesh_nodes: ArrayLike,
    mesh_elements: ArrayLike,
    probe_orientation: str = "RAS",
    probe_units: str = "mm",
    embedding_step: float = 0.1,
    max_embedding_steps: int = 1_000,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Register fNIRS source and detector positions to a tetrahedral head mesh.

    Parameters
    ----------
    source_coordinates : array-like
        Source coordinates with shape ``(n_sources, 3)``.
    detector_coordinates : array-like
        Detector coordinates with shape ``(n_detectors, 3)``.
    mesh_nodes : array-like
        Mesh node coordinates with shape ``(n_nodes, 3)``.
    mesh_elements : array-like
        Tetrahedral vertex indices with shape ``(n_elements, 4)``. Both zero-based
        and one-based indices are accepted.
    probe_orientation : str, default="RAS"
        Three-letter orientation code describing the probe coordinate system.
    probe_units : {"mm", "m"}, default="mm"
        Unit used by the probe coordinates. Mesh coordinates are assumed to be
        millimetres.
    embedding_step : float, default=0.1
        Distance in millimetres by which exterior optodes move toward the mesh
        center during each embedding iteration.
    max_embedding_steps : int, default=1000
        Maximum number of embedding iterations before registration fails.
    plot : bool, default=False
        Whether to create a diagnostic 3D registration plot.

    Returns
    -------
    registered_sources : numpy.ndarray
        Registered source coordinates.
    registered_detectors : numpy.ndarray
        Registered detector coordinates.
    source_directions : numpy.ndarray
        Unit vectors pointing from each source toward the mesh center.
    detector_directions : numpy.ndarray
        Unit vectors pointing from each detector toward the mesh center.
    source_elements : numpy.ndarray
        Zero-based indices of tetrahedra containing the sources.
    detector_elements : numpy.ndarray
        Zero-based indices of tetrahedra containing the detectors.

    Raises
    ------
    ValueError
        If an input shape, orientation, unit, or embedding setting is invalid.
    RuntimeError
        If one or more optodes cannot be embedded within ``max_embedding_steps``.
    """
    sources = _as_coordinate_array(source_coordinates, "source_coordinates")
    detectors = _as_coordinate_array(detector_coordinates, "detector_coordinates")
    nodes = _as_coordinate_array(mesh_nodes, "mesh_nodes")
    elements = _as_element_array(mesh_elements, nodes.shape[0])
    if probe_units not in {"mm", "m"}:
        raise ValueError("probe_units must be either 'mm' or 'm'")
    if embedding_step <= 0:
        raise ValueError("embedding_step must be positive")
    if max_embedding_steps < 0:
        raise ValueError("max_embedding_steps must be non-negative")

    orientation_matrices = make_orientation_matrices()
    try:
        orientation_matrix = orientation_matrices[probe_orientation.upper()]
    except KeyError as error:
        raise ValueError(f"Unknown probe orientation {probe_orientation!r}") from error

    unit_scale = 1_000.0 if probe_units == "m" else 1.0
    sources_ras = unit_scale * sources @ orientation_matrix.T
    detectors_ras = unit_scale * detectors @ orientation_matrix.T

    optodes_ras = np.vstack((sources_ras, detectors_ras))
    mesh_center = (nodes.min(axis=0) + nodes.max(axis=0)) / 2.0
    probe_center = (optodes_ras.min(axis=0) + optodes_ras.max(axis=0)) / 2.0
    roughly_aligned = optodes_ras + mesh_center - probe_center

    _, registered_optodes, _ = trimesh.registration.icp(roughly_aligned, nodes)
    registered_sources = registered_optodes[: sources.shape[0]]
    registered_detectors = registered_optodes[sources.shape[0] :]

    source_directions = find_optode_directions(registered_sources, nodes)
    detector_directions = find_optode_directions(registered_detectors, nodes)
    registered_sources, source_elements = _embed_optodes(
        registered_sources,
        source_directions,
        nodes,
        elements,
        embedding_step,
        max_embedding_steps,
    )
    registered_detectors, detector_elements = _embed_optodes(
        registered_detectors,
        detector_directions,
        nodes,
        elements,
        embedding_step,
        max_embedding_steps,
    )

    if plot:
        _plot_registration(
            nodes,
            registered_sources,
            registered_detectors,
            source_directions,
            detector_directions,
        )

    return (
        registered_sources,
        registered_detectors,
        source_directions,
        detector_directions,
        source_elements,
        detector_elements,
    )


def find_optode_directions(optode_coordinates: ArrayLike, mesh_nodes: ArrayLike) -> np.ndarray:
    """Compute inward unit directions from optodes toward the mesh center.

    Parameters
    ----------
    optode_coordinates : array-like
        Optode coordinates with shape ``(n_optodes, 3)``.
    mesh_nodes : array-like
        Mesh node coordinates with shape ``(n_nodes, 3)``.

    Returns
    -------
    numpy.ndarray
        Unit direction vectors with shape ``(n_optodes, 3)``.

    Raises
    ------
    ValueError
        If an optode lies exactly at the mesh center.
    """
    optodes = _as_coordinate_array(optode_coordinates, "optode_coordinates")
    nodes = _as_coordinate_array(mesh_nodes, "mesh_nodes")
    mesh_center = (nodes.min(axis=0) + nodes.max(axis=0)) / 2.0
    directions = mesh_center - optodes
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError("Cannot determine a direction for an optode at the mesh center")
    return directions / lengths


def _embed_optodes(
    coordinates: np.ndarray,
    directions: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    step: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    embedded_coordinates = coordinates.copy()
    containing_elements = _find_containing_elements(embedded_coordinates, nodes, elements)
    for _ in range(max_steps):
        exterior_mask = containing_elements < 0
        if not np.any(exterior_mask):
            return embedded_coordinates, containing_elements
        embedded_coordinates[exterior_mask] += directions[exterior_mask] * step
        containing_elements[exterior_mask] = _find_containing_elements(
            embedded_coordinates[exterior_mask],
            nodes,
            elements,
        )

    if np.any(containing_elements < 0):
        number_exterior = int(np.count_nonzero(containing_elements < 0))
        raise RuntimeError(f"Failed to embed {number_exterior} optode(s) within {max_steps} steps")
    return embedded_coordinates, containing_elements


def _plot_registration(
    mesh_nodes: np.ndarray,
    source_coordinates: np.ndarray,
    detector_coordinates: np.ndarray,
    source_directions: np.ndarray,
    detector_directions: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.scatter(*mesh_nodes.T, s=0.5, alpha=0.1, color="peru")
    axes.quiver(*source_coordinates.T, *source_directions.T, color="red", length=15)
    axes.quiver(*detector_coordinates.T, *detector_directions.T, color="blue", length=15)
    axes.view_init(elev=30, azim=45)
