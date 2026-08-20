"""Utilities for preparing and visualizing registered fNIRS probes."""

from collections.abc import Mapping
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal

import h5py
import numpy as np
import trimesh
from numpy.typing import ArrayLike

from mmc_nirs.utils.mesh_utils import (
    _find_containing_elements,
    as_coordinate_array,
    as_element_array,
    make_orientation_matrices,
    make_surface_mesh,
)
from mmc_nirs.utils.prepared_input_io import require_config_section, require_fields

if TYPE_CHECKING:
    from matplotlib.figure import Figure

PREPARED_PROBE_KEYS = {
    "sourcepos",
    "detpos",
    "sourcedir",
    "detnorms",
    "source_elements",
    "detector_elements",
    "channel_pairings",
}
_PROBE_SETTINGS_KEYS = {
    "probe_units",
    "probe_orientation",
    "short_separation_flag",
    "short_separation_arg",
    "embedding_step",
    "max_embedding_steps",
}


def validate_probe_settings(experiment_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate probe configuration and return canonical registration values."""
    probe_settings = require_config_section(experiment_config, "probe_settings", _PROBE_SETTINGS_KEYS)

    units = probe_settings["probe_units"]
    if not isinstance(units, str) or units.lower() not in {"mm", "cm", "m"}:
        raise ValueError("probe_units must be either 'mm', 'cm', or 'm'")

    orientation = probe_settings["probe_orientation"]
    try:
        make_orientation_matrices()[orientation.upper()]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"Unknown probe orientation {orientation!r}") from error

    embedding_step = probe_settings["embedding_step"]
    if (
        not isinstance(embedding_step, (int, float, np.integer, np.floating))
        or isinstance(embedding_step, (bool, np.bool_))
        or not np.isfinite(embedding_step)
        or embedding_step <= 0
    ):
        raise ValueError("embedding_step must be a finite positive scalar")

    max_embedding_steps = probe_settings["max_embedding_steps"]
    if (
        not isinstance(max_embedding_steps, (int, np.integer))
        or isinstance(max_embedding_steps, (bool, np.bool_))
        or max_embedding_steps < 0
    ):
        raise ValueError("max_embedding_steps must be a non-negative integer")

    short_separation_flag = probe_settings["short_separation_flag"]
    short_separation_arg = probe_settings["short_separation_arg"]
    if not isinstance(short_separation_flag, str):
        raise ValueError("short_separation_flag must be either 'distance' or 'index'")
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

    return {
        "probe_units": units.lower(),
        "probe_orientation": orientation.upper(),
        "short_separation_flag": normalized_flag,
        "short_separation_arg": short_separation_arg,
        "embedding_step": embedding_step,
        "max_embedding_steps": max_embedding_steps,
    }


def as_channel_pairing_array(values: ArrayLike, name: str = "channel_pairings") -> np.ndarray:
    """Return a copied integer array containing source-detector index pairs."""
    pairings = np.asarray(values)
    if pairings.ndim != 2 or pairings.shape[0] == 0 or pairings.shape[1] != 2:
        raise ValueError(f"{name} must be a non-empty array with shape (n_channels, 2)")
    if not np.issubdtype(pairings.dtype, np.integer):
        if not np.all(np.isfinite(pairings)) or not np.all(pairings == np.floor(pairings)):
            raise ValueError(f"{name} must contain integer indices")
    return pairings.astype(np.intp, copy=True)


def _normalize_pairing_indices(
    indices: np.ndarray,
    size: int,
    coordinate_type: str,
    *,
    index_base: Literal["auto", "zero", "one"] = "auto",
) -> np.ndarray:
    """Return zero-based channel indices after validating their range."""
    if index_base not in {"auto", "zero", "one"}:
        raise ValueError("index_base must be 'auto', 'zero', or 'one'")
    normalized = np.asarray(indices, dtype=np.intp).copy()
    if index_base == "one" or (index_base == "auto" and normalized.min() >= 1 and normalized.max() <= size):
        normalized -= 1
    if normalized.min() < 0 or normalized.max() >= size:
        raise ValueError(f"channel_pairings contains an out-of-range {coordinate_type} index")
    return normalized


def normalize_channel_pairings(
    values: ArrayLike,
    source_count: int,
    detector_count: int,
    *,
    index_base: Literal["auto", "zero", "one"] = "auto",
) -> np.ndarray:
    """Validate source-detector pairings and return canonical zero-based indices."""
    pairings = as_channel_pairing_array(values)
    pairings[:, 0] = _normalize_pairing_indices(pairings[:, 0], source_count, "source", index_base=index_base)
    pairings[:, 1] = _normalize_pairing_indices(pairings[:, 1], detector_count, "detector", index_base=index_base)
    return pairings


def as_optode_element_indices(
    values: ArrayLike,
    optode_count: int,
    element_count: int,
    name: str,
) -> np.ndarray:
    """Return one validated zero-based containing-element index per optode."""
    indices = np.asarray(values)
    if indices.shape != (optode_count,):
        raise ValueError(f"{name} must contain one value per optode")
    if not np.issubdtype(indices.dtype, np.integer):
        if not np.all(np.isfinite(indices)) or not np.all(indices == np.floor(indices)):
            raise ValueError(f"{name} must contain integer indices")
    indices = indices.astype(np.intp, copy=True)
    if indices.min() < 0 or indices.max() >= element_count:
        raise ValueError(f"{name} contains an out-of-range element index")
    return indices


def as_unit_direction_array(values: ArrayLike, name: str) -> np.ndarray:
    """Return finite three-dimensional unit vectors required by MMC."""
    directions = as_coordinate_array(values, name)
    if not np.all(np.isclose(np.linalg.norm(directions, axis=1), 1.0, rtol=1e-6, atol=1e-8)):
        raise ValueError(f"{name} must contain unit-length vectors; prepare_probe may not have been run")
    return directions


def validate_prepared_probe(
    prepared_probe: Mapping[str, ArrayLike],
    element_count: int,
) -> dict[str, np.ndarray]:
    """Validate and copy the canonical arrays in a prepared probe mapping."""
    if not isinstance(prepared_probe, Mapping):
        raise TypeError("prepared_probe must be a mapping")
    require_fields(prepared_probe, PREPARED_PROBE_KEYS, "prepared_probe")

    source_positions = as_coordinate_array(prepared_probe["sourcepos"], "prepared_probe['sourcepos']")
    detector_positions = as_coordinate_array(prepared_probe["detpos"], "prepared_probe['detpos']")
    source_directions = as_coordinate_array(prepared_probe["sourcedir"], "prepared_probe['sourcedir']")
    detector_directions = as_coordinate_array(prepared_probe["detnorms"], "prepared_probe['detnorms']")
    if source_directions.shape != source_positions.shape:
        raise ValueError("prepared_probe['sourcedir'] must match sourcepos")
    if detector_directions.shape != detector_positions.shape:
        raise ValueError("prepared_probe['detnorms'] must match detpos")
    source_directions = as_unit_direction_array(source_directions, "prepared_probe['sourcedir']")
    detector_directions = as_unit_direction_array(detector_directions, "prepared_probe['detnorms']")

    source_elements = as_optode_element_indices(
        prepared_probe["source_elements"],
        len(source_positions),
        element_count,
        "prepared_probe['source_elements']",
    )
    detector_elements = as_optode_element_indices(
        prepared_probe["detector_elements"],
        len(detector_positions),
        element_count,
        "prepared_probe['detector_elements']",
    )
    channel_pairings = normalize_channel_pairings(
        prepared_probe["channel_pairings"],
        len(source_positions),
        len(detector_positions),
        index_base="zero",
    )
    return {
        "sourcepos": source_positions,
        "detpos": detector_positions,
        "sourcedir": source_directions,
        "detnorms": detector_directions,
        "source_elements": source_elements,
        "detector_elements": detector_elements,
        "channel_pairings": channel_pairings,
    }


def flatten_channel_pairings(
    channel_pairings: ArrayLike,
    source_count: int,
    detector_count: int,
) -> np.ndarray:
    """Return unique zero-based rows in a source-major source-detector matrix."""
    pairings = normalize_channel_pairings(
        channel_pairings,
        source_count,
        detector_count,
        index_base="zero",
    )
    flattened = pairings[:, 0] * detector_count + pairings[:, 1]
    return np.unique(flattened)


def _signed_surface_distances(
    coordinates: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    surface: trimesh.Trimesh | None = None,
) -> np.ndarray:
    """Return surface distances that are positive outside and negative inside."""
    if surface is None:
        surface = make_surface_mesh(nodes, elements)
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
) -> "Figure":
    """Plot registered optodes, channel pairings, and signed surface distances."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    nodes = as_coordinate_array(mesh_nodes, "mesh_nodes")
    elements = as_element_array(mesh_elements, len(nodes))
    source_color = "#d1495b"
    detector_color = "#0077b6"
    channel_color = "#546a7b"
    mesh_color = "#d9a066"
    surface = make_surface_mesh(nodes, elements)
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
    return figure


def load_channel_pairs_from_snirf(snirf_file: str | PathLike[str]) -> np.ndarray:
    """Load unique source-detector pairings in first-measurement order."""
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
    pairings = np.asarray(pairs, dtype=int).reshape(-1, 2)
    _, first_indices = np.unique(pairings, axis=0, return_index=True)
    return pairings[np.sort(first_indices)]
