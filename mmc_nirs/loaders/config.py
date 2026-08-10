"""Experiment configuration loading."""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

EXPERIMENTS_PACKAGE = "mmc_nirs.experiments"


def _experiment_resource(experiment: str):
    if not experiment or Path(experiment).name != experiment:
        raise ValueError("experiment must be a non-empty name, not a path")
    return files(EXPERIMENTS_PACKAGE).joinpath(experiment)


def load_config(experiment: str) -> dict[str, Any]:
    """Load a bundled experiment configuration.

    Parameters
    ----------
    experiment : str
        Name of the experiment directory, for example ``"pain"``.

    Returns
    -------
    dict[str, Any]
        Parsed contents of the experiment's ``config.json`` file.

    Raises
    ------
    ValueError
        If ``experiment`` is empty or contains path components.
    FileNotFoundError
        If the requested experiment does not contain a configuration file.
    json.JSONDecodeError
        If the configuration is not valid JSON.
    """
    config_resource = _experiment_resource(experiment).joinpath("config.json")
    if not config_resource.is_file():
        raise FileNotFoundError(f"No configuration found for experiment {experiment!r}")

    with config_resource.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration for experiment {experiment!r} must contain a JSON object")
    return config
