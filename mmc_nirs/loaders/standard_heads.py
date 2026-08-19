"""Download standard-head data from the project Hugging Face dataset."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download

__all__ = ["load_standard_head"]

_REPOSITORY = "nielsbracher/fnirs-simbox-assets"
_REPO_TYPE = "dataset"
_TOKEN_PATH = Path(__file__).resolve().parents[2] / "tokens" / "HF_TOKEN.txt"


@dataclass(frozen=True)
class _StandardHead:
    remote_path: PurePosixPath
    files: tuple[str, ...]


_STANDARD_HEADS = {
    "colin27": _StandardHead(
        remote_path=PurePosixPath("standard_heads/colin27"),
        files=("README.md", "colin27_mesh.npz", "orientation.txt", "segmentation_map.npz"),
    )
}


def _read_token() -> str:
    try:
        token = _TOKEN_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise RuntimeError(f"Hugging Face token file not found: {_TOKEN_PATH}") from error
    if not token:
        raise RuntimeError(f"Hugging Face token file is empty: {_TOKEN_PATH}")
    return token


def _standard_head(name: str) -> _StandardHead:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("standard_head must be a non-empty supported name, not a path")
    try:
        return _STANDARD_HEADS[name]
    except KeyError as error:
        supported = ", ".join(sorted(_STANDARD_HEADS))
        raise ValueError(f"Unknown standard head {name!r}; supported heads: {supported}") from error


def _missing_files(directory: Path, standard_head: _StandardHead) -> list[str]:
    return [filename for filename in standard_head.files if not (directory / filename).is_file()]


def load_standard_head(
    name: str,
    *,
    save: bool = False,
    directory: str | PathLike[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Return a local directory containing one supported standard head.

    With ``save=False``, the requested Hugging Face subtree remains in the
    Hugging Face cache. With ``save=True``, it is copied to ``directory/name``;
    ``directory`` defaults to the current working directory. A complete saved
    directory is reused unless ``overwrite`` is true.

    The project dataset is currently private, so authentication uses the same
    repository-local ``tokens/HF_TOKEN.txt`` convention as the MMC runtime.
    """
    if not isinstance(save, bool):
        raise TypeError("save must be a boolean")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")

    standard_head = _standard_head(name)
    destination = (Path.cwd() if directory is None else Path(directory).expanduser()) / name
    if save and not overwrite and not _missing_files(destination, standard_head):
        return destination

    snapshot_root = Path(
        snapshot_download(
            repo_id=_REPOSITORY,
            repo_type=_REPO_TYPE,
            allow_patterns=f"{standard_head.remote_path.as_posix()}/**",
            token=_read_token(),
            force_download=overwrite,
        )
    )
    cached_directory = snapshot_root.joinpath(*standard_head.remote_path.parts)
    missing = _missing_files(cached_directory, standard_head)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Downloaded standard head {name!r} is missing: {missing_list}")

    if not save:
        return cached_directory

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cached_directory, destination, dirs_exist_ok=True)
    missing = _missing_files(destination, standard_head)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Saved standard head {name!r} is missing: {missing_list}")
    return destination
