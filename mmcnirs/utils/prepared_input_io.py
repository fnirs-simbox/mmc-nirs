"""Path resolution and archive I/O for prepared light-transport inputs."""

from collections.abc import Collection, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np


def require_fields(
    available_fields: Collection[str],
    required_fields: Collection[str],
    description: str,
) -> None:
    """Raise when a named collection omits one or more required fields."""
    missing_fields = set(required_fields).difference(available_fields)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"{description} is missing required field(s): {missing}")


def require_config_section(
    experiment_config: Mapping[str, Any],
    section_name: str,
    required_fields: Collection[str],
) -> Mapping[str, Any]:
    """Return a configuration section after requiring all standardized fields."""
    required = sorted(required_fields)
    required_text = ", ".join(required)
    section = experiment_config.get(section_name)
    if not isinstance(section, Mapping):
        raise ValueError(
            f"experiment_config[{section_name!r}] must be a mapping; "
            f"required keys: {required_text}; missing keys: {required_text}"
        )

    missing = sorted(set(required).difference(section))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"experiment_config[{section_name!r}] required keys: {required_text}; missing keys: {missing_text}"
        )
    return section


def resolve_prepared_input_path(experiment_config: Mapping[str, Any], filename_key: str) -> Path:
    """Resolve a configured prepared-input archive path."""
    filepaths = experiment_config.get("filepaths")
    if not isinstance(filepaths, Mapping):
        raise ValueError("experiment_config must contain a 'filepaths' mapping")

    experiment_dir = experiment_config.get("experiment_dir")
    if not isinstance(experiment_dir, (str, PathLike)):
        raise ValueError("experiment_config['experiment_dir'] must be a path")

    filename = filepaths.get(filename_key)
    if not isinstance(filename, (str, PathLike)):
        raise ValueError(f"filepaths[{filename_key!r}] must be a path")
    configured_filename = Path(filename)
    if configured_filename.suffix.lower() != ".npz":
        raise ValueError(f"filepaths[{filename_key!r}] must name a .npz file")
    return Path(experiment_dir).expanduser() / configured_filename


def load_npz_archive(path: Path, required_keys: set[str]) -> dict[str, np.ndarray]:
    """Load an NPZ archive and require the specified fields."""
    with np.load(path, allow_pickle=False) as archive:
        require_fields(archive.files, required_keys, f"NPZ archive {path}")
        return {key: archive[key].copy() for key in archive.files}


def save_npz_archive(path: Path, values: Mapping[str, Any]) -> None:
    """Create the output directory and save values in an NPZ archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)
