"""Compare the current probe preparation with the legacy pain-data result."""

from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

# Allow this script to be run directly from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mmc_nirs.light_transport.prepare_jacobian_probe import (  # noqa: E402
    load_channel_pairs_from_snirf,
    prepare_jacobian_probe,
)
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
    """Load source and detector positions from the MATLAB probe structure."""
    probe = loadmat(path, squeeze_me=True, struct_as_record=False)["SD"]
    return np.asarray(probe.SrcPos, dtype=float), np.asarray(probe.DetPos, dtype=float)


def load_prepared_mesh(path: Path) -> dict[str, np.ndarray]:
    """Load and normalize the legacy prepared-mesh archive."""
    with np.load(path, allow_pickle=False) as archive:
        return {
            "nodes": archive["nodes"][:, :3].copy(),
            "elements": archive["elem"][:, :4].copy(),
        }


def load_legacy_probe(path: Path, mesh_nodes: np.ndarray) -> dict[str, np.ndarray]:
    """Load the legacy probe and convert its left-right axis to the current mesh frame.

    The complete legacy orientation workflow maps ``LIA`` inputs to RAS in the
    same way as the current workflow, but the saved ground-truth archive uses
    the opposite left-right direction. Positions are therefore reflected about
    the mesh's X midpoint, while direction vectors require only an X sign flip.
    """
    with np.load(path, allow_pickle=False) as archive:
        legacy_probe = {key: archive[key].copy() for key in archive.files}

    mesh_x_midpoint = (mesh_nodes[:, 0].min() + mesh_nodes[:, 0].max()) / 2.0
    for position_key in ("sourcepos", "detpos"):
        legacy_probe[position_key][:, 0] = 2.0 * mesh_x_midpoint - legacy_probe[position_key][:, 0]
    legacy_probe["detnorms"][:, 0] *= -1.0
    return legacy_probe


def print_position_comparison(name: str, current: np.ndarray, legacy: np.ndarray) -> None:
    """Print pointwise Euclidean-error statistics for a coordinate array."""
    if current.shape != legacy.shape:
        raise ValueError(f"{name} shape differs: current={current.shape}, legacy={legacy.shape}")
    errors = np.linalg.norm(current - legacy, axis=1)
    print(f"{name}: mean={errors.mean():.6f} mm, RMSE={np.sqrt(np.mean(errors**2)):.6f} mm, max={errors.max():.6f} mm")


def print_direction_comparison(name: str, current: np.ndarray, legacy: np.ndarray) -> None:
    """Print vector and angular-error statistics for a direction array."""
    if current.shape != legacy.shape:
        raise ValueError(f"{name} shape differs: current={current.shape}, legacy={legacy.shape}")
    vector_errors = np.linalg.norm(current - legacy, axis=1)
    current_unit = current / np.linalg.norm(current, axis=1, keepdims=True)
    legacy_unit = legacy / np.linalg.norm(legacy, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.sum(current_unit * legacy_unit, axis=1), -1.0, 1.0)))
    print(
        f"{name}: mean vector error={vector_errors.mean():.6f}, "
        f"max vector error={vector_errors.max():.6f}, "
        f"mean angle={angles.mean():.6f} degrees, max angle={angles.max():.6f} degrees"
    )


def main() -> None:
    """Prepare the pain-data probe and compare it with the legacy archive."""
    mesh_path = resolve_prepared_input_path(EXPERIMENT_CONFIG, "meshfile")
    output_path = resolve_prepared_input_path(EXPERIMENT_CONFIG, "probefile")
    source_positions, detector_positions = load_probe_positions(DATA_DIRECTORY / "probe.SD")
    prepared_mesh = load_prepared_mesh(mesh_path)
    channel_pairings = load_channel_pairs_from_snirf(DATA_DIRECTORY / "FingerTapping.snirf")

    current_probe = prepare_jacobian_probe(
        source_positions=source_positions,
        detector_positions=detector_positions,
        prepared_mesh=prepared_mesh,
        units="mm",
        orientation="LIA",
        channel_pairings=channel_pairings,
        short_separation_flag="distance",
        short_separation_arg=20.0,
        experiment_config=EXPERIMENT_CONFIG,
        overwrite=True,
    )

    legacy_probe = load_legacy_probe(DATA_DIRECTORY / "legacy_probe.npz", prepared_mesh["nodes"])
    print_position_comparison("sourcepos", current_probe["sourcepos"], legacy_probe["sourcepos"])
    print_position_comparison("detpos", current_probe["detpos"], legacy_probe["detpos"])
    print_direction_comparison("detnorms", current_probe["detnorms"], legacy_probe["detnorms"])

    print(f"Current prepared probe saved to {output_path}")


if __name__ == "__main__":
    main()
