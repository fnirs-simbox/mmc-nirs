"""Precomputed MMC forward models and loading tools for SimNIRS."""

from importlib.metadata import version

from .loaders import load_config, load_default_config, load_mmc_files

__version__ = version("mmc-nirs")
