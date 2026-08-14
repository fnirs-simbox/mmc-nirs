"""Path resolution and archive I/O for prepared light-transport inputs."""

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np


def resolve_prepared_input_path(experiment_config: Mapping[str, Any], filename_key: str) -> Path:
    """Resolve a configured prepared-input archive path."""
    filepaths = experiment_config.get("filepaths")
    if not isinstance(filepaths, Mapping):
        raise ValueError("experiment_config must contain a 'filepaths' mapping")

    experiment_dir = filepaths.get("experiment_dir")
    if not isinstance(experiment_dir, (str, PathLike)):
        raise ValueError("filepaths['experiment_dir'] must be a path")

    filename = filepaths.get(filename_key)
    if not isinstance(filename, (str, PathLike)):
        raise ValueError(f"filepaths[{filename_key!r}] must be a path")
    configured_filename = Path(filename)
    if configured_filename.suffix.lower() != ".npz":
        raise ValueError(f"filepaths[{filename_key!r}] must name a .npz file")
    return Path(experiment_dir).expanduser() / configured_filename


def load_prepared_input(path: Path, required_keys: set[str]) -> dict[str, np.ndarray]:
    """Load a prepared archive and require its canonical fields."""
    with np.load(path, allow_pickle=False) as archive:
        missing_keys = required_keys.difference(archive.files)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Prepared archive {path} is missing required field(s): {missing}")
        return {key: archive[key].copy() for key in archive.files}


def save_prepared_input(path: Path, prepared: Mapping[str, Any]) -> None:
    """Create the configured output directory and save a prepared archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **prepared)
