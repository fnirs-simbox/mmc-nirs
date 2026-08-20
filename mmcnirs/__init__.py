"""Light-transport preparation and loading tools for SimNIRS."""

from importlib.metadata import version

from .loaders import load_config, load_default_config, load_light_transport_results, load_standard_head

__version__ = version("mmcnirs")
