"""Prepare and plot the pattern-cutting probe registration."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Allow this script to be run directly from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mmc_nirs.light_transport.prepare_jacobian_probe import prepare_jacobian_probe  # noqa: E402
from mmc_nirs.light_transport.probe_utils import load_channel_pairs_from_snirf  # noqa: E402
from mmc_nirs.utils.prepared_input_io import resolve_prepared_input_path  # noqa: E402


DATA_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_CONFIG = {
    "filepaths": {
        "experiment_dir": DATA_DIRECTORY,
        "meshfile": "mesh.npz",
        "probefile": "current_probe.npz",
    }
}


def load_probe_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load source and detector positions from the MATLAB probe archive."""
    probe = loadmat(path, squeeze_me=True, struct_as_record=False)
    return np.asarray(probe["sourcepos"], dtype=float), np.asarray(probe["detpos"], dtype=float)


def load_prepared_mesh(path: Path) -> dict[str, np.ndarray]:
    """Load and normalize the legacy prepared-mesh archive."""
    with np.load(path, allow_pickle=False) as archive:
        return {
            "nodes": archive["nodes"][:, :3].copy(),
            "elements": archive["elem"][:, :4].copy(),
        }


def main() -> None:
    """Prepare the pattern-cutting probe and save its diagnostic figure."""
    mesh_path = resolve_prepared_input_path(EXPERIMENT_CONFIG, "meshfile")
    probe_output_path = resolve_prepared_input_path(EXPERIMENT_CONFIG, "probefile")
    figure_output_path = DATA_DIRECTORY / "registered_probe_diagnostic.png"

    source_positions, detector_positions = load_probe_positions(DATA_DIRECTORY / "probe.mat")
    prepared_mesh = load_prepared_mesh(mesh_path)
    channel_pairings = load_channel_pairs_from_snirf(DATA_DIRECTORY / "NIRS-2019-08-10_006.snirf")

    prepare_jacobian_probe(
        source_positions=source_positions,
        detector_positions=detector_positions,
        prepared_mesh=prepared_mesh,
        units="mm",
        orientation="RAS",
        channel_pairings=channel_pairings,
        short_separation_flag="distance",
        short_separation_arg=14.0,
        experiment_config=EXPERIMENT_CONFIG,
        plot=True,
        overwrite=True,
    )

    figure = plt.gcf()
    figure.savefig(
        figure_output_path,
        dpi=220,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    print(f"Prepared probe saved to {probe_output_path}")
    print(f"Diagnostic figure saved to {figure_output_path}")


if __name__ == "__main__":
    main()
