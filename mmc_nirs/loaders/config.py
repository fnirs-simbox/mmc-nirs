"""Experiment configuration loading."""

import json
from os import PathLike
from pathlib import Path
from typing import Any

from .assets import ensure_experiment_assets


def _read_config(config_path, description: str) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration {description} must contain a JSON object")

    filepaths = config.get("filepaths")
    if not isinstance(filepaths, dict):
        raise ValueError(f"Configuration {description} must contain a 'filepaths' object")
    return config


def load_default_config(experiment: str) -> dict[str, Any]:
    """Load a bundled experiment configuration.

    Parameters
    ----------
    experiment : str
        Name of the experiment directory, for example ``"pain"``.

    Returns
    -------
    dict[str, Any]
        Parsed configuration with the bundled experiment directory attached.

    Raises
    ------
    ValueError
        If ``experiment`` is empty or contains path components.
    FileNotFoundError
        If the requested experiment does not contain a configuration file.
    json.JSONDecodeError
        If the configuration is not valid JSON.
    """
    assets_directory = ensure_experiment_assets(experiment)
    config_path = assets_directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No configuration found for experiment {experiment!r}")

    config = _read_config(config_path, f"for experiment {experiment!r}")
    config["filepaths"]["experiment_dir"] = assets_directory
    return config


def load_config(config_path: str | PathLike[str]) -> dict[str, Any]:
    """Load an experiment configuration from a JSON file.

    Relative experiment directories are resolved from the directory containing
    the configuration file. If ``experiment_dir`` is absent or empty, the data
    files are assumed to be stored alongside ``config.json``.

    Parameters
    ----------
    config_path : str or path-like
        Path to an experiment's JSON configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration with a resolved experiment directory.

    Raises
    ------
    FileNotFoundError
        If the configuration does not exist.
    json.JSONDecodeError
        If the configuration is not valid JSON.
    ValueError
        If the configuration is not a JSON object or has invalid ``filepaths``.
    """
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No configuration found at {path}")

    path = path.resolve()
    config = _read_config(path, f"at {path}")
    configured_directory = config["filepaths"].get("experiment_dir", "")
    if not isinstance(configured_directory, (str, PathLike)):
        raise ValueError("'experiment_dir' must be a path")

    experiment_dir = Path(configured_directory).expanduser()
    if not experiment_dir.is_absolute():
        experiment_dir = path.parent / experiment_dir
    experiment_dir = experiment_dir.resolve()

    config["filepaths"]["experiment_dir"] = experiment_dir
    return config
