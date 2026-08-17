"""Utilities for preparing and visualizing registered fNIRS probes."""

from os import PathLike

import h5py
import numpy as np
import trimesh
from numpy.typing import ArrayLike

from mmc_nirs.utils.mesh_utils import _as_coordinate_array, _as_element_array, _find_containing_elements

from .register_probe import _make_surface_mesh


def _pairing_indices(indices: np.ndarray, size: int, coordinate_type: str) -> np.ndarray:
    """Return zero-based channel indices after validating their range."""
    if indices.min() >= 1 and indices.max() <= size:
        return indices - 1
    if indices.min() < 0 or indices.max() >= size:
        raise ValueError(f"channel_pairings contains an out-of-range {coordinate_type} index")
    return indices


def _signed_surface_distances(
    coordinates: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    surface: trimesh.Trimesh | None = None,
) -> np.ndarray:
    """Return surface distances that are positive outside and negative inside."""
    if surface is None:
        surface = _make_surface_mesh(nodes, elements)
    _, distances, _ = trimesh.proximity.closest_point_naive(surface, coordinates)
    inside = _find_containing_elements(coordinates, nodes, elements) >= 0
    return np.where(inside, -distances, distances)


def _plot_probe_registration(
    mesh_nodes: ArrayLike,
    mesh_elements: ArrayLike,
    source_coordinates: np.ndarray,
    detector_coordinates: np.ndarray,
    source_directions: np.ndarray,
    detector_directions: np.ndarray,
    source_indices: np.ndarray,
    detector_indices: np.ndarray,
) -> None:
    """Plot registered optodes, channel pairings, and signed surface distances."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    nodes = _as_coordinate_array(mesh_nodes, "mesh_nodes")
    elements = _as_element_array(mesh_elements, len(nodes))
    source_color = "#d1495b"
    detector_color = "#0077b6"
    channel_color = "#546a7b"
    mesh_color = "#d9a066"
    surface = _make_surface_mesh(nodes, elements)
    optode_coordinates = np.vstack((source_coordinates, detector_coordinates))
    surface_distances = _signed_surface_distances(optode_coordinates, nodes, elements, surface)

    figure = plt.figure(figsize=(18, 7), layout="constrained", facecolor="#fafafa")
    grid = figure.add_gridspec(1, 2, width_ratios=(1.2, 1.0))
    probe_axes = figure.add_subplot(grid[0, 0], projection="3d")
    distance_axes = figure.add_subplot(grid[0, 1])

    mesh_collection = Poly3DCollection(
        nodes[surface.faces],
        facecolor=mesh_color,
        edgecolor="none",
        alpha=0.08,
        rasterized=True,
    )
    probe_axes.add_collection3d(mesh_collection)

    unique_pairings = np.unique(np.column_stack((source_indices, detector_indices)), axis=0)
    for source_index, detector_index in unique_pairings:
        channel = np.vstack((source_coordinates[source_index], detector_coordinates[detector_index]))
        probe_axes.plot(*channel.T, color=channel_color, linewidth=0.7, alpha=0.28)

    probe_axes.scatter(
        *source_coordinates.T,
        s=45,
        color=source_color,
        marker="o",
        depthshade=False,
        zorder=4,
    )
    probe_axes.scatter(
        *detector_coordinates.T,
        s=38,
        color=detector_color,
        marker="^",
        depthshade=False,
        zorder=4,
    )
    probe_axes.quiver(
        *source_coordinates.T,
        *source_directions.T,
        color=source_color,
        length=8,
        linewidth=0.8,
        alpha=0.7,
        normalize=True,
    )
    probe_axes.quiver(
        *detector_coordinates.T,
        *detector_directions.T,
        color=detector_color,
        length=8,
        linewidth=0.8,
        alpha=0.7,
        normalize=True,
    )

    lower = nodes.min(axis=0)
    upper = nodes.max(axis=0)
    padding = 0.04 * (upper - lower)
    probe_axes.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    probe_axes.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    probe_axes.set_zlim(lower[2] - padding[2], upper[2] + padding[2])
    probe_axes.set_box_aspect(upper - lower)
    probe_axes.set_proj_type("ortho")
    probe_axes.view_init(elev=24, azim=-68)
    probe_axes.set_title("Registered probe and channel pairings", pad=12, fontsize=12, fontweight="bold")
    probe_axes.set_xlabel("R-L (mm)", labelpad=5)
    probe_axes.set_ylabel("A-P (mm)", labelpad=5)
    probe_axes.set_zlabel("I-S (mm)", labelpad=5)
    probe_axes.grid(False)
    for axis in (probe_axes.xaxis, probe_axes.yaxis, probe_axes.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#dddddd")

    number_sources = len(source_coordinates)
    labels = [f"S{index}" for index in range(1, number_sources + 1)]
    labels.extend(f"D{index}" for index in range(1, len(detector_coordinates) + 1))
    bar_colors = [source_color] * number_sources + [detector_color] * len(detector_coordinates)
    optode_indices = np.arange(len(optode_coordinates))
    distance_axes.bar(optode_indices, surface_distances, color=bar_colors, width=0.78)
    distance_axes.set_xticks(optode_indices, labels, rotation=75, ha="right", fontsize=7)
    distance_axes.set_xlabel("Optode (S = source, D = detector)")
    distance_axes.set_ylabel("Signed distance to mesh surface (mm)")
    displayed_minimum = 0.0 if np.isclose(surface_distances.min(), 0.0, atol=5e-4) else surface_distances.min()
    displayed_maximum = 0.0 if np.isclose(surface_distances.max(), 0.0, atol=5e-4) else surface_distances.max()
    distance_axes.set_title(
        "Signed optode distance to mesh surface\n"
        "positive = outside, negative = inside\n"
        f"mean={surface_distances.mean():.3f} mm, "
        f"range=[{displayed_minimum:.3f}, {displayed_maximum:.3f}] mm",
        fontsize=12,
        fontweight="bold",
    )
    distance_axes.set_axisbelow(True)
    distance_axes.grid(axis="y", color="#dddddd", linewidth=0.8)
    distance_axes.axhline(0.0, color="#555555", linewidth=1.0)
    distance_axes.spines[["top", "right"]].set_visible(False)
    distance_axes.legend(
        handles=[
            Line2D([0], [0], color=source_color, linewidth=7, label="Source"),
            Line2D([0], [0], color=detector_color, linewidth=7, label="Detector"),
            Line2D([0], [0], color=channel_color, linewidth=1.5, label="Channel pairing"),
        ],
        frameon=False,
    )

    figure.suptitle("Probe registration diagnostic", fontsize=16, fontweight="bold")


def load_channel_pairs_from_snirf(snirf_file: str | PathLike[str]) -> np.ndarray:
    """Load source-detector channel pairings from a SNIRF file."""
    with h5py.File(snirf_file, "r") as snirf:
        data_group = snirf["nirs"]["data1"]
        measurement_keys = sorted(
            (key for key in data_group if key.startswith("measurementList")),
            key=lambda key: int(key.removeprefix("measurementList")),
        )
        pairs = [
            [
                int(data_group[key]["sourceIndex"][()].item()),
                int(data_group[key]["detectorIndex"][()].item()),
            ]
            for key in measurement_keys
        ]
    return np.asarray(pairs, dtype=int)
