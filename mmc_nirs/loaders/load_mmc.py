"""Load bundled MMC forward-model arrays."""

from importlib.resources import as_file
from typing import Any

import numpy as np

from .config import _experiment_resource, load_config


def load_mmc_files(experiment: str, use_jacobian: bool = True) -> dict[str, Any]:
    """Load the mesh, registered probe, and Jacobians for an experiment.

    Parameters
    ----------
    experiment : str
        Name of a bundled experiment.
    use_jacobian : bool, default=True
        Whether to load Jacobians, baseline measurements, channel indices, and the
        activation map.

    Returns
    -------
    dict[str, Any]
        Arrays and metadata required to initialize a SimNIRS simulator.

    Raises
    ------
    ValueError
        If Jacobian files use inconsistent channel indices or contain invalid
        one-based indices.
    FileNotFoundError
        If the experiment or one of its configured data files does not exist.
    """
    config = load_config(experiment)
    file_paths = config["filepaths"]

    with as_file(_experiment_resource(experiment)) as experiment_directory:
        with np.load(experiment_directory / file_paths["meshfile"]) as mesh_archive:
            nodes = mesh_archive[file_paths["nodes_var"]].copy()

        with np.load(experiment_directory / file_paths["probefile"]) as probe_archive:
            source_positions = probe_archive["sourcepos"].copy()
            detector_positions = probe_archive["detpos"].copy()
            detector_norms = probe_archive["detnorms"].copy()

        jacobian_list: list[np.ndarray] = []
        measurements_zero_list: list[np.ndarray] = []
        channel_indices: np.ndarray | None = None

        if use_jacobian:
            for jacobian_file in file_paths.get("jacobians", []):
                with np.load(experiment_directory / jacobian_file) as jacobian_archive:
                    current_indices = np.asarray(jacobian_archive["channelidx"]).reshape(-1)
                    if not np.issubdtype(current_indices.dtype, np.integer) or np.any(current_indices < 1):
                        raise ValueError(f"{jacobian_file} contains invalid one-based channel indices")
                    current_indices = current_indices.astype(np.intp, copy=False) - 1

                    if channel_indices is None:
                        channel_indices = current_indices
                    elif not np.array_equal(channel_indices, current_indices):
                        raise ValueError("All Jacobian files must use the same channel indices")

                    jacobian = jacobian_archive["J"]
                    measurements_zero = jacobian_archive["mea0"]
                    jacobian_list.append(jacobian[current_indices, :].copy())
                    measurements_zero_list.append(measurements_zero[current_indices].reshape(-1, 1).copy())

            activation_file = file_paths.get("activation_map")
            activation_map = (
                np.load(experiment_directory / activation_file, allow_pickle=False) if activation_file else None
            )
        else:
            activation_map = None

    return {
        "source_positions": source_positions,
        "detector_positions": detector_positions,
        "detector_norms": detector_norms,
        "nodes": nodes,
        "jacobian_list": jacobian_list,
        "measurements_zero_list": measurements_zero_list,
        "channel_idx": channel_indices,
        "activation_map": activation_map,
        "use_jacobian": use_jacobian,
    }
