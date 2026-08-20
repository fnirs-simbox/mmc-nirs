"""Download standard-head data from the project Hugging Face dataset."""

from __future__ import annotations

import shutil
from os import PathLike
from pathlib import Path

from .hf_loader import HF_RESOURCE_KEYWORDS, download_hf_resource, required_hf_files

__all__ = ["load_standard_head"]


def _validate_standard_head(name: str) -> None:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("standard_head must be a non-empty supported name, not a path")
    supported_heads = HF_RESOURCE_KEYWORDS["standard-head"]
    if name not in supported_heads:
        supported = ", ".join(supported_heads)
        raise ValueError(f"Unknown standard head {name!r}; supported heads: {supported}")


def _missing_files(directory: Path, filenames: tuple[str, ...]) -> list[str]:
    return [filename for filename in filenames if not (directory / filename).is_file()]


def load_standard_head(
    name: str,
    *,
    save: bool = False,
    directory: str | PathLike[str] | None = None,
    assets_root: str | PathLike[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Return a local directory containing one supported standard head.

    With ``save=False``, the requested subtree is returned from the central
    ``./mmcnirs-assets`` download root, which can be changed with
    ``assets_root``. With ``save=True``, it is also copied to
    ``directory/name``; ``directory`` defaults to the current working
    directory. A complete saved directory is reused unless ``overwrite`` is
    true.

    Downloads use the public project dataset and require no Hugging Face token.
    """
    if not isinstance(save, bool):
        raise TypeError("save must be a boolean")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")

    _validate_standard_head(name)
    required_files = required_hf_files("standard-head", name)
    destination = (Path.cwd() if directory is None else Path(directory).expanduser()) / name
    if save and not overwrite and not _missing_files(destination, required_files):
        return destination

    downloaded_directory = download_hf_resource(
        "standard-head",
        name,
        assets_root=assets_root,
        force_download=overwrite,
    )
    missing = _missing_files(downloaded_directory, required_files)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Downloaded standard head {name!r} is missing: {missing_list}")

    if not save:
        return downloaded_directory

    if destination.resolve() == downloaded_directory.resolve():
        return downloaded_directory

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(downloaded_directory, destination, dirs_exist_ok=True)
    missing = _missing_files(destination, required_files)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Saved standard head {name!r} is missing: {missing_list}")
    return destination
