"""Register fNIRS probes to tetrahedral head meshes."""

import numpy as np
import trimesh
from numpy.typing import ArrayLike
from scipy.optimize import minimize

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
    probe_units : {"mm", "cm", "m"}, default="mm"
        Unit used by the probe coordinates. Mesh coordinates are assumed to be
        millimetres.
    embedding_step : float, default=0.1
        Distance in millimetres by which exterior optodes move toward the mesh
        center during each embedding iteration.
    max_embedding_steps : int, default=1000
        Maximum number of embedding iterations before registration fails.
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
        If translation optimization fails or one or more optodes cannot be
        embedded within ``max_embedding_steps``.
    """
    # Validate and normalize coordinate and element arrays. Element indices are
    # converted to zero-based indexing by _as_element_array when necessary.
    sources = _as_coordinate_array(source_coordinates, "source_coordinates")
    detectors = _as_coordinate_array(detector_coordinates, "detector_coordinates")
    nodes = _as_coordinate_array(mesh_nodes, "mesh_nodes")
    elements = _as_element_array(mesh_elements, nodes.shape[0])

    # Convert the probe's declared length unit to the mesh's millimeter unit.
    unit_scales = {"mm": 1.0, "cm": 10.0, "m": 1_000.0}
    try:
        unit_scale = unit_scales[probe_units.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("probe_units must be either 'mm', 'cm', or 'm'") from error

    # Reject embedding settings that cannot move optodes toward the mesh.
    if embedding_step <= 0:
        raise ValueError("embedding_step must be positive")
    if max_embedding_steps < 0:
        raise ValueError("max_embedding_steps must be non-negative")

    # Look up the matrix that maps the probe coordinate convention to RAS.
    orientation_matrices = make_orientation_matrices()
    try:
        orientation_matrix = orientation_matrices[probe_orientation.upper()]
    except KeyError as error:
        raise ValueError(f"Unknown probe orientation {probe_orientation!r}") from error

    # Reorient the sources and detectors and express them in millimeters.
    sources_ras = unit_scale * sources @ orientation_matrix.T
    detectors_ras = unit_scale * detectors @ orientation_matrix.T

    # Combine all optodes so registration applies exactly the same transform to
    # sources and detectors, preserving their relative arrangement.
    optodes_ras = np.vstack((sources_ras, detectors_ras))

    # Produce a stable initial placement: center the probe over the mesh in X
    # and Y, then align the probe's highest Z coordinate with the mesh's top.
    mesh_center = (nodes.min(axis=0) + nodes.max(axis=0)) / 2.0
    probe_center = (optodes_ras.min(axis=0) + optodes_ras.max(axis=0)) / 2.0
    alignment_offset = mesh_center - probe_center
    alignment_offset[2] = nodes[:, 2].max() - optodes_ras[:, 2].max()
    roughly_aligned = optodes_ras + alignment_offset

    # Refine the rough placement using translation only, minimizing the mean
    # squared distance between the optodes and the exterior mesh surface.
    registered_optodes = _minimize_surface_translation(roughly_aligned, nodes, elements)

    # Restore separate source and detector arrays after their shared registration.
    registered_sources = registered_optodes[: sources.shape[0]]
    registered_detectors = registered_optodes[sources.shape[0] :]

    # Calculate fixed inward directions from each registered optode toward the
    # center of the mesh; these directions drive the embedding step below.
    source_directions = find_optode_directions(registered_sources, nodes)
    detector_directions = find_optode_directions(registered_detectors, nodes)

    # Move any exterior sources inward until each lies in a tetrahedron.
    registered_sources, source_elements = _embed_optodes(
        registered_sources,
        source_directions,
        nodes,
        elements,
        embedding_step,
        max_embedding_steps,
    )

    # Perform the same embedding and containing-element lookup for detectors.
    registered_detectors, detector_elements = _embed_optodes(
        registered_detectors,
        detector_directions,
        nodes,
        elements,
        embedding_step,
        max_embedding_steps,
    )

    # Return final coordinates, inward directions, and containing tetrahedra in
    # separate source and detector arrays expected by downstream simulations.
    return (
        registered_sources,
        registered_detectors,
        source_directions,
        detector_directions,
        source_elements,
        detector_elements,
    )


def _minimize_surface_translation(
    coordinates: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
) -> np.ndarray:
    """Translate optodes to minimize their squared distances to the mesh surface.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Optode coordinates with shape ``(n_optodes, 3)``.
    nodes : numpy.ndarray
        Tetrahedral mesh-node coordinates with shape ``(n_nodes, 3)``.
    elements : numpy.ndarray
        Zero-based tetrahedral vertex indices with shape ``(n_elements, 4)``.

    Returns
    -------
    numpy.ndarray
        Translated optode coordinates with shape ``(n_optodes, 3)``. A single
        translation vector is applied to every optode, preserving their relative
        positions and orientation.

    Raises
    ------
    RuntimeError
        If the translation optimization does not converge successfully.
    """
    surface = _make_surface_mesh(nodes, elements)

    def mean_squared_surface_distance(translation: np.ndarray) -> float:
        """Return the mean squared distance from translated optodes to the mesh surface.

        Parameters
        ----------
        translation : numpy.ndarray
            Three-dimensional translation vector applied to every optode.

        Returns
        -------
        float
            Mean of the squared shortest distances from the translated optodes
            to the triangular mesh surface.
        """
        translated_coordinates = coordinates + translation
        _, distances, _ = trimesh.proximity.closest_point_naive(surface, translated_coordinates)
        return float(np.mean(np.square(distances)))

    result = minimize(mean_squared_surface_distance, np.zeros(3), method="Powell")
    if not result.success:
        raise RuntimeError(f"Failed to optimize probe translation: {result.message}")
    return coordinates + result.x


def _make_surface_mesh(nodes: np.ndarray, elements: np.ndarray) -> trimesh.Trimesh:
    """Construct the exterior triangular surface of a tetrahedral mesh."""
    tetrahedron_faces = elements[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
    _, unique_indices, face_counts = np.unique(
        np.sort(tetrahedron_faces, axis=1),
        axis=0,
        return_index=True,
        return_counts=True,
    )
    boundary_faces = tetrahedron_faces[unique_indices[face_counts == 1]]
    return trimesh.Trimesh(vertices=nodes, faces=boundary_faces, process=False)


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
