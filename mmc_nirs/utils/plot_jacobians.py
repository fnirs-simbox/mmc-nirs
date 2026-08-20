"""Visualize node-wise Jacobian sensitivity on a prepared head mesh."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from numpy.typing import ArrayLike

from mmc_nirs.utils.jacobian_utils import validate_jacobian
from mmc_nirs.utils.mesh_utils import validate_prepared_mesh
from mmc_nirs.utils.probe_utils import flatten_channel_pairings, validate_prepared_probe


def plot_tissue_sensitivity(
    prepared_mesh: Mapping[str, ArrayLike],
    prepared_probe: Mapping[str, ArrayLike],
    jacobian: ArrayLike,
    channel_selection: int | Literal["all"],
    save_directory: str | Path,
    *,
    tissue_type: int = 2,
    save_filename: str = "tissue_sensitivity.png",
    outer_color: str = "lightgray",
    tissue_color: str = "#d1495b",
    outer_alpha: float = 0.03,
    max_tissue_alpha: float = 0.9,
    alpha_threshold: float = 0.0,
    upper_percentile: float = 99.5,
    use_absolute_values: bool = True,
    optode_size: float = 40,
    channel_color: str = "#546a7b",
    channel_linewidth: float = 1.5,
    channel_alpha: float = 0.8,
    elev: float = 17,
    azim: float = -225,
) -> Figure:
    """Plot and save sensitivity for one configured channel or their mean.

    ``channel_selection`` is either a zero-based row in the prepared probe's
    ``channel_pairings`` or ``"all"``. The latter averages the Jacobian rows
    belonging to all configured channels and draws every configured pairing.
    The Jacobian itself must contain every source-detector combination in
    source-major order, as returned by :func:`generate_jacobian`.
    """
    mesh = validate_prepared_mesh(prepared_mesh)
    probe = validate_prepared_probe(prepared_probe, len(mesh["elements"]))

    nodes = mesh["nodes"]
    elements = mesh["elements"]
    tissue_types = mesh["element_tissue_ids"]
    source_coordinates = probe["sourcepos"]
    detector_coordinates = probe["detpos"]
    configured_pairings = probe["channel_pairings"]
    jacobian_array = validate_jacobian(
        jacobian,
        len(source_coordinates),
        len(detector_coordinates),
        len(nodes),
    )

    node_values, channel_pairings = _select_channel_values(
        jacobian_array,
        configured_pairings,
        channel_selection,
        len(source_coordinates),
        len(detector_coordinates),
    )

    values = np.abs(node_values) if use_absolute_values else node_values
    element_values = values[elements].mean(axis=1)

    outer_faces, _ = find_boundary_faces_with_parent(elements)

    tissue_mask = tissue_types == tissue_type
    if not np.any(tissue_mask):
        raise ValueError(f"No elements found with tissue type {tissue_type}.")

    tissue_element_indices = np.flatnonzero(tissue_mask)
    tissue_faces, tissue_parent_local = find_boundary_faces_with_parent(elements[tissue_mask])
    tissue_parent_global = tissue_element_indices[tissue_parent_local]
    tissue_face_values = element_values[tissue_parent_global]

    positive = element_values[element_values > 0]
    if len(positive) == 0:
        raise ValueError("All element values are zero.")

    vmax = np.percentile(positive, upper_percentile)
    normalized_values = np.clip(tissue_face_values / vmax, 0.0, 1.0)

    outer_rgba = np.tile(to_rgba(outer_color), (len(outer_faces), 1))
    outer_rgba[:, 3] = outer_alpha

    source_indices = channel_pairings[:, 0]
    detector_indices = channel_pairings[:, 1]
    channel_segments = np.stack(
        (source_coordinates[source_indices], detector_coordinates[detector_indices]),
        axis=1,
    )

    figure = plt.figure(figsize=(16, 8), layout="constrained", facecolor="#fafafa")
    axes = [
        figure.add_subplot(1, 2, 1, projection="3d"),
        figure.add_subplot(1, 2, 2, projection="3d"),
    ]
    alpha_gammas = (0.25, 1.0)

    xlim = (nodes[:, 0].min(), nodes[:, 0].max())
    ylim = (nodes[:, 1].min(), nodes[:, 1].max())
    zlim = (nodes[:, 2].min(), nodes[:, 2].max())
    box_aspect = np.ptp(nodes, axis=0)

    for axis, alpha_gamma in zip(axes, alpha_gammas, strict=True):
        normalized_alpha = normalized_values**alpha_gamma
        normalized_alpha[normalized_values < alpha_threshold] = 0.0
        tissue_alpha = max_tissue_alpha * normalized_alpha

        tissue_rgba = np.tile(to_rgba(tissue_color), (len(tissue_faces), 1))
        tissue_rgba[:, 3] = tissue_alpha

        outer_collection = Poly3DCollection(
            nodes[outer_faces],
            facecolors=outer_rgba,
            edgecolor="none",
            rasterized=True,
        )
        axis.add_collection3d(outer_collection)

        tissue_collection = Poly3DCollection(
            nodes[tissue_faces],
            facecolors=tissue_rgba,
            edgecolor="none",
            rasterized=True,
        )
        axis.add_collection3d(tissue_collection)

        channel_collection = Line3DCollection(
            channel_segments,
            colors=channel_color,
            linewidths=channel_linewidth,
            alpha=channel_alpha,
        )
        axis.add_collection3d(channel_collection)

        axis.scatter(
            *source_coordinates.T,
            s=optode_size,
            c="red",
            depthshade=False,
            edgecolors="black",
            linewidths=0.4,
        )
        axis.scatter(
            *detector_coordinates.T,
            s=optode_size,
            c="blue",
            depthshade=False,
            edgecolors="black",
            linewidths=0.4,
        )

        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_zlim(*zlim)
        axis.set_box_aspect(box_aspect)
        axis.view_init(elev=elev, azim=azim)
        axis.set_axis_off()
        axis.set_title(rf"$\gamma = {alpha_gamma}$", fontsize=14)

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="red",
                markeredgecolor="black",
                markersize=8,
                label="Source",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="blue",
                markeredgecolor="black",
                markersize=8,
                label="Detector",
            ),
            Line2D(
                [0],
                [0],
                color=channel_color,
                linewidth=channel_linewidth,
                label="Channel",
            ),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
    )

    output_directory = Path(save_directory).expanduser()
    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(f"save_directory is not a directory: {output_directory}")

    filename = Path(save_filename)
    if filename.name != save_filename or filename.suffix.lower() != ".png":
        raise ValueError("save_filename must be a PNG filename without directory components")

    output_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_directory / filename)
    return figure


def _select_channel_values(
    jacobian: np.ndarray,
    configured_pairings: np.ndarray,
    channel_selection: int | Literal["all"],
    source_count: int,
    detector_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected node values and the pairings to draw."""
    if channel_selection == "all":
        channel_rows = flatten_channel_pairings(
            configured_pairings,
            source_count,
            detector_count,
        )
        unique_pairings = np.column_stack(np.divmod(channel_rows, detector_count))
        return jacobian[channel_rows].mean(axis=0), unique_pairings

    if isinstance(channel_selection, (int, np.integer)) and not isinstance(
        channel_selection,
        (bool, np.bool_),
    ):
        channel_index = int(channel_selection)
        if channel_index < 0 or channel_index >= len(configured_pairings):
            raise ValueError(
                "channel_selection must be a zero-based index into "
                "prepared_probe['channel_pairings']"
            )
        selected_pairing = configured_pairings[[channel_index]]
        source_index, detector_index = selected_pairing[0]
        jacobian_row = source_index * detector_count + detector_index
        return jacobian[jacobian_row], selected_pairing

    raise ValueError("channel_selection must be a zero-based integer or 'all'")


def find_boundary_faces_with_parent(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary triangles and the tetrahedron that owns each one."""
    all_faces = elements[
        :,
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
    ].reshape(-1, 3)
    parent_elements = np.repeat(np.arange(len(elements)), 4)
    sorted_faces = np.sort(all_faces, axis=1)
    _, first_indices, counts = np.unique(
        sorted_faces,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    boundary_occurrences = first_indices[counts == 1]
    return all_faces[boundary_occurrences], parent_elements[boundary_occurrences]
