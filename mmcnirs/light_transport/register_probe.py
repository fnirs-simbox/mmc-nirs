"""Register fNIRS probes to tetrahedral head meshes."""

from typing import Any

import numpy as np
import trimesh
from numpy import dtype, ndarray
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from mmcnirs.utils.mesh_utils import (
    _find_containing_elements,
    as_coordinate_array,
    as_element_array,
    make_orientation_matrices,
    make_surface_mesh,
)

from mmcnirs.utils.probe_utils import find_smoothed_surface_directions


def register_probe(
    source_coordinates: ArrayLike,
    detector_coordinates: ArrayLike,
    mesh_nodes: ArrayLike,
    mesh_elements: ArrayLike,
    probe_orientation: str = "RAS",
    probe_units: str = "mm",
    embedding_step: float = 1e-3,
) -> tuple[
    Any, Any, ndarray | int, ndarray | int, ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]]
]:
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
    embedding_step : float, default=1e-3 [mm]
        Distance in millimetres inside mesh of optodes wrt the mesh surface.
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
    # converted to zero-based indexing by as_element_array when necessary.
    sources = as_coordinate_array(source_coordinates, "source_coordinates")
    detectors = as_coordinate_array(detector_coordinates, "detector_coordinates")
    nodes = as_coordinate_array(mesh_nodes, "mesh_nodes")
    elements = as_element_array(mesh_elements, nodes.shape[0])

    # Convert the probe's declared length unit to the mesh's millimeter unit.
    unit_scales = {"mm": 1.0, "cm": 10.0, "m": 1_000.0}
    try:
        unit_scale = unit_scales[probe_units.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("probe_units must be either 'mm', 'cm', or 'm'") from error

    # Reject embedding settings that cannot move optodes toward the mesh.
    if embedding_step <= 0:
        raise ValueError("embedding_step must be positive")

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

    # calculate surface
    surface = make_surface_mesh(nodes, elements)

    # Refine the rough placement using translation only, minimizing the mean
    # squared distance between the optodes and the exterior mesh surface.
    registered_optodes = _minimize_surface_translation(roughly_aligned, surface)

    (
        embedded_optodes,
        surface_points,
        embedding_directions,
        containing_elements,
        surface_distances,
    ) = project_optodes_just_inside(
        registered_optodes,
        surface,
        nodes,
        elements,
        embedding_step,
    )

    n_sources = len(sources)

    registered_sources = embedded_optodes[:n_sources]
    registered_detectors = embedded_optodes[n_sources:]

    source_surface_points = surface_points[:n_sources]
    detector_surface_points = surface_points[n_sources:]

    source_elements = containing_elements[:n_sources]
    detector_elements = containing_elements[n_sources:]

    source_surface_directions, _ = find_smoothed_surface_directions(
        surface,
        source_surface_points,
        nodes,
    )
    detector_surface_directions, _ = find_smoothed_surface_directions(
        surface,
        detector_surface_points,
        nodes,
    )

    return (
        registered_sources,
        registered_detectors,
        source_surface_directions,
        detector_surface_directions,
        source_elements,
        detector_elements,
    )


def _minimize_surface_translation(
    coordinates: np.ndarray,
    surface: trimesh.Trimesh,
) -> np.ndarray:
    """Translate optodes to minimize their squared distances to the mesh surface.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Optode coordinates with shape ``(n_optodes, 3)``.
    surface : numpy.ndarray
        Exterior triangular surface of a tetrahedral mesh.

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


def project_optodes_just_inside(coordinates, surface, nodes, elements, epsilon_mm=1e-3):
    surface_points, distances, face_ids = trimesh.proximity.closest_point_naive(surface, coordinates)
    containing_before = _find_containing_elements(coordinates, nodes, elements)
    delta = surface_points - coordinates
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths == 0):
        raise RuntimeError("Cannot determine projection direction for optode exactly on surface")
    directions = delta / lengths[:, None]
    # If already inside, surface_points - coordinates points outward.
    inside = containing_before >= 0
    directions[inside] *= -1.0
    embedded = surface_points + epsilon_mm * directions
    containing_after = _find_containing_elements(embedded, nodes, elements)

    if np.any(containing_after < 0):
        raise RuntimeError("Some projected optodes are not inside after epsilon embedding")

    return embedded, surface_points, directions, containing_after, distances
