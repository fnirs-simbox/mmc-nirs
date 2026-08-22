"""Load canonical light-transport inputs used by the downstream fNIRS simulator."""

from contextlib import contextmanager
from importlib.resources import as_file
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

_MESH_FIELDS = (
    "nodes",
    "elements",
    "element_tissue_ids",
    "ordered_tissue_ids",
    "ordered_tissues",
)
_PROBE_FIELDS = (
    "sourcepos",
    "detpos",
    "sourcedir",
    "detnorms",
    "source_elements",
    "detector_elements",
    "channel_pairings",
    "short_separation_indices",
    "long_separation_indices",
)
_JACOBIAN_FIELDS = ("J", "mea0", "channelidx")


@contextmanager
def _experiment_directory(configured_directory):
    if isinstance(configured_directory, (str, PathLike)):
        directory = Path(configured_directory).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"Experiment directory does not exist: {directory}")
        yield directory
        return

    with as_file(configured_directory) as directory:
        yield directory


def _load_fields(archive_path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(archive_path, allow_pickle=False) as archive:
        return {field: archive[field].copy() for field in fields}


def load_light_transport_results(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """Load canonical mesh, probe, Jacobian, and segmentation dictionaries.

    This function only deserializes the prepared light-transport inputs.
    Numerical, shape, and cross-file validation is performed by SimNIRS when
    constructing a Jacobian-based ``TrialSimulator``.

    Parameters
    ----------
    experiment_config : dict
        Experiment configuration containing ``experiment_dir``, ``wavelengths``,
        and the mesh, probe, Jacobian, and optional segmentation-map file paths.

    Returns
    -------
    dict[str, Any]
        Canonical ``mesh``, ``probe``, ``jacobians``, and ``segmentation_map``
        dictionaries accepted by SimNIRS.

    Raises
    ------
    FileNotFoundError
        If the experiment directory or one of its configured data files does
        not exist.
    KeyError
        If the configuration or an archive omits a required field.
    ValueError
        If the wavelength and Jacobian file lists have different lengths.
    """
    file_paths = experiment_config["filepaths"]
    configured_directory = experiment_config["experiment_dir"]

    with _experiment_directory(configured_directory) as experiment_directory:
        mesh = _load_fields(experiment_directory / file_paths["meshfile"], _MESH_FIELDS)
        probe = _load_fields(experiment_directory / file_paths["probefile"], _PROBE_FIELDS)

        wavelengths = experiment_config["wavelengths"]
        jacobian_files = file_paths["jacobians"]
        if len(wavelengths) != len(jacobian_files):
            raise ValueError("wavelengths and Jacobian file paths must have the same length")

        jacobians: dict[int, dict[str, np.ndarray]] = {}
        for wavelength, jacobian_file in zip(wavelengths, jacobian_files, strict=True):
            jacobians[int(wavelength)] = _load_fields(
                experiment_directory / jacobian_file,
                _JACOBIAN_FIELDS,
            )

        segmentation_path = file_paths.get("segmentation_map", "segmentation_map.npz")
        with np.load(experiment_directory / segmentation_path, allow_pickle=False) as segmentation_archive:
            segmentation_map = {region: segmentation_archive[region].copy() for region in segmentation_archive.files}

    return {
        "mesh": mesh,
        "probe": probe,
        "jacobians": jacobians,
        "segmentation_map": segmentation_map,
    }
