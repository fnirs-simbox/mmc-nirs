"""Visualize the current and legacy pain-data probe registrations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from verify_prepare_probe import load_legacy_probe


DATA_DIRECTORY = Path(__file__).resolve().parent
SOURCE_COLOR = "#d1495b"
DETECTOR_COLOR = "#0077b6"
MESH_COLOR = "#d9a066"
CHANNEL_COLOR = "#546a7b"


def boundary_faces(elements: np.ndarray) -> np.ndarray:
    """Return the exterior triangular faces of a tetrahedral mesh."""
    tetrahedron_faces = elements[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
    _, unique_indices, counts = np.unique(
        np.sort(tetrahedron_faces, axis=1),
        axis=0,
        return_index=True,
        return_counts=True,
    )
    return tetrahedron_faces[unique_indices[counts == 1]]


def zero_based_unique_pairings(pairings: np.ndarray, number_sources: int, number_detectors: int) -> np.ndarray:
    """Normalize source-detector pairings and remove wavelength duplicates."""
    pairs = np.unique(np.asarray(pairings, dtype=np.intp), axis=0)
    if pairs[:, 0].min() >= 1 and pairs[:, 0].max() <= number_sources:
        pairs[:, 0] -= 1
    if pairs[:, 1].min() >= 1 and pairs[:, 1].max() <= number_detectors:
        pairs[:, 1] -= 1
    return pairs


def add_mesh_surface(axes, nodes: np.ndarray, faces: np.ndarray) -> None:
    """Add a translucent head-mesh surface to a 3D axes."""
    surface = Poly3DCollection(
        nodes[faces],
        facecolor=MESH_COLOR,
        edgecolor="none",
        alpha=0.08,
        rasterized=True,
    )
    axes.add_collection3d(surface)


def add_channels(
    axes,
    sources: np.ndarray,
    detectors: np.ndarray,
    pairings: np.ndarray,
) -> None:
    """Draw a light line for each unique source-detector channel."""
    for source_index, detector_index in pairings:
        channel = np.vstack((sources[source_index], detectors[detector_index]))
        axes.plot(*channel.T, color=CHANNEL_COLOR, linewidth=0.7, alpha=0.28)


def add_probe(
    axes,
    sources: np.ndarray,
    detectors: np.ndarray,
    pairings: np.ndarray,
    detector_directions: np.ndarray,
) -> None:
    """Draw channel connections, optodes, and detector directions."""
    add_channels(axes, sources, detectors, pairings)
    axes.scatter(*sources.T, s=45, color=SOURCE_COLOR, marker="o", depthshade=False, zorder=4)
    axes.scatter(*detectors.T, s=38, color=DETECTOR_COLOR, marker="^", depthshade=False, zorder=4)
    axes.quiver(
        *detectors.T,
        *detector_directions.T,
        color=DETECTOR_COLOR,
        length=8,
        linewidth=0.8,
        alpha=0.7,
        normalize=True,
    )


def style_axes(axes, nodes: np.ndarray, title: str) -> None:
    """Apply consistent limits, proportions, and camera styling."""
    lower = nodes.min(axis=0)
    upper = nodes.max(axis=0)
    padding = 0.04 * (upper - lower)
    axes.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    axes.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    axes.set_zlim(lower[2] - padding[2], upper[2] + padding[2])
    axes.set_box_aspect(upper - lower)
    axes.set_proj_type("ortho")
    axes.view_init(elev=24, azim=-68)
    axes.set_title(title, pad=12, fontsize=12, fontweight="bold")
    axes.set_xlabel("R–L (mm)", labelpad=5)
    axes.set_ylabel("A–P (mm)", labelpad=5)
    axes.set_zlabel("I–S (mm)", labelpad=5)
    axes.grid(False)
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#dddddd")


def create_visualization(output_path: Path, show: bool = True) -> None:
    """Create and save the current-versus-legacy registration figure."""
    current_path = DATA_DIRECTORY / "current_probe.npz"
    if not current_path.is_file():
        raise FileNotFoundError(f"{current_path} does not exist; run verify_prepare_probe.py first")

    with np.load(DATA_DIRECTORY / "mesh.npz", allow_pickle=False) as mesh:
        nodes = mesh["nodes"][:, :3].copy()
        elements = mesh["elem"][:, :4].astype(np.intp, copy=True)
    if elements.min() == 1 and elements.max() == len(nodes):
        elements -= 1
    faces = boundary_faces(elements)

    with np.load(current_path, allow_pickle=False) as archive:
        current = {key: archive[key].copy() for key in archive.files}
    legacy = load_legacy_probe(DATA_DIRECTORY / "legacy_probe.npz", nodes)

    pairings = zero_based_unique_pairings(
        current["channel_pairings"],
        len(current["sourcepos"]),
        len(current["detpos"]),
    )
    source_errors = np.linalg.norm(current["sourcepos"] - legacy["sourcepos"], axis=1)
    detector_errors = np.linalg.norm(current["detpos"] - legacy["detpos"], axis=1)

    figure = plt.figure(figsize=(19, 7), layout="constrained", facecolor="#fafafa")
    axes = [figure.add_subplot(1, 3, index, projection="3d") for index in range(1, 4)]
    for current_axes in axes:
        current_axes.set_facecolor("#fafafa")
        add_mesh_surface(current_axes, nodes, faces)

    add_probe(
        axes[0],
        current["sourcepos"],
        current["detpos"],
        pairings,
        current["detnorms"],
    )
    style_axes(axes[0], nodes, "Current translation-only registration")

    add_probe(
        axes[1],
        legacy["sourcepos"],
        legacy["detpos"],
        pairings,
        legacy["detnorms"],
    )
    style_axes(axes[1], nodes, "Legacy registration")

    for current_point, legacy_point in zip(current["sourcepos"], legacy["sourcepos"], strict=True):
        axes[2].plot(*np.vstack((legacy_point, current_point)).T, color=SOURCE_COLOR, alpha=0.35, linewidth=0.8)
    for current_point, legacy_point in zip(current["detpos"], legacy["detpos"], strict=True):
        axes[2].plot(*np.vstack((legacy_point, current_point)).T, color=DETECTOR_COLOR, alpha=0.3, linewidth=0.8)
    axes[2].scatter(
        *legacy["sourcepos"].T,
        s=42,
        facecolors="white",
        edgecolors=SOURCE_COLOR,
        marker="o",
        depthshade=False,
    )
    axes[2].scatter(
        *legacy["detpos"].T,
        s=38,
        facecolors="white",
        edgecolors=DETECTOR_COLOR,
        marker="^",
        depthshade=False,
    )
    axes[2].scatter(*current["sourcepos"].T, s=24, color=SOURCE_COLOR, marker="o", depthshade=False)
    axes[2].scatter(*current["detpos"].T, s=22, color=DETECTOR_COLOR, marker="^", depthshade=False)
    style_axes(
        axes[2],
        nodes,
        f"Corresponding optodes\nmean error: sources {source_errors.mean():.1f} mm, "
        f"detectors {detector_errors.mean():.1f} mm",
    )

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SOURCE_COLOR,
            markeredgecolor=SOURCE_COLOR,
            label="Source",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=DETECTOR_COLOR,
            markeredgecolor=DETECTOR_COLOR,
            label="Detector",
        ),
        Line2D([0], [0], color=CHANNEL_COLOR, linewidth=1.0, alpha=0.6, label="Unique channel"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", label="Legacy (overlay)"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#555555", label="Current (overlay)"),
    ]
    figure.legend(handles=legend_items, loc="lower center", ncol=5, frameon=False)
    figure.suptitle("Pain-data probe registration comparison", fontsize=16, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(f"Visualization saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    """Parse command-line options and create the visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIRECTORY / "registered_probe_comparison.png",
        help="output image path (default: registered_probe_comparison.png in pain_data)",
    )
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    arguments = parser.parse_args()
    create_visualization(arguments.output, show=not arguments.no_show)


if __name__ == "__main__":
    main()
