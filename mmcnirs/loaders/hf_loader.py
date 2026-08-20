"""Centralized anonymous downloads from the project Hugging Face dataset."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Literal

from huggingface_hub import hf_hub_download, snapshot_download

__all__ = [
    "DEFAULT_ASSETS_DIRECTORY",
    "HF_RESOURCE_KEYWORDS",
    "download_hf_resource",
    "required_hf_files",
]

DEFAULT_ASSETS_DIRECTORY = "mmcnirs-assets"
_REPOSITORY = "nielsbracher/fnirs-simbox-assets"
_REPO_TYPE = "dataset"
_REVISION = "main"


@dataclass(frozen=True)
class _HfResource:
    remote_path: PurePosixPath
    kind: Literal["directory", "file", "dynamic-file"]
    required_files: tuple[str, ...] = ()


_RESOURCES = {
    ("experiment", "pain"): _HfResource(
        remote_path=PurePosixPath("experiments/pain"),
        kind="directory",
        required_files=(
            "README.md",
            "config.json",
            "jacobian_690.npz",
            "jacobian_830.npz",
            "mesh.npz",
            "probe.npz",
            "segmentation_map.npz",
        ),
    ),
    ("experiment", "pattern_cutting"): _HfResource(
        remote_path=PurePosixPath("experiments/pattern_cutting"),
        kind="directory",
        required_files=(
            "README.md",
            "config.json",
            "jacobian_760.npz",
            "jacobian_850.npz",
            "mesh.npz",
            "probe.npz",
            "segmentation_map.npz",
        ),
    ),
    ("standard-head", "colin27"): _HfResource(
        remote_path=PurePosixPath("standard-heads/colin27"),
        kind="directory",
        required_files=("README.md", "colin27_mesh.npz", "orientation.txt", "segmentation_map.npz"),
    ),
    ("workflow", "e2e-files"): _HfResource(
        remote_path=PurePosixPath("e2e-files"),
        kind="directory",
        required_files=("FingerTapping.snirf", "README.md", "config.json", "optical_properties.json", "probe.SD"),
    ),
    ("runtime", "manifest"): _HfResource(
        remote_path=PurePosixPath("mmc-runtime/manifest.json"),
        kind="file",
    ),
    ("runtime", "archive"): _HfResource(
        remote_path=PurePosixPath("mmc-runtime"),
        kind="dynamic-file",
    ),
}

HF_RESOURCE_KEYWORDS = {
    category: tuple(keyword for resource_category, keyword in _RESOURCES if resource_category == category)
    for category in dict.fromkeys(category for category, _ in _RESOURCES)
}


def _resource(category: str, keyword: str) -> _HfResource:
    if category not in HF_RESOURCE_KEYWORDS:
        supported = ", ".join(HF_RESOURCE_KEYWORDS)
        raise ValueError(f"Unknown Hugging Face resource category {category!r}; supported categories: {supported}")
    try:
        return _RESOURCES[(category, keyword)]
    except KeyError as error:
        supported = ", ".join(HF_RESOURCE_KEYWORDS[category])
        raise ValueError(f"Unknown {category} resource {keyword!r}; supported keywords: {supported}") from error


def _resolved_assets_root(assets_root: str | PathLike[str] | None) -> Path:
    root = Path.cwd() / DEFAULT_ASSETS_DIRECTORY if assets_root is None else Path(assets_root).expanduser()
    return root.resolve()


def _validated_dynamic_path(resource: _HfResource, path_in_repo: str | None) -> PurePosixPath:
    if not isinstance(path_in_repo, str) or not path_in_repo:
        raise ValueError("A non-empty path_in_repo is required for a dynamic Hugging Face resource")
    remote_path = PurePosixPath(path_in_repo)
    if (
        remote_path.is_absolute()
        or any(part in {".", ".."} for part in remote_path.parts)
        or remote_path == resource.remote_path
        or not remote_path.is_relative_to(resource.remote_path)
    ):
        raise ValueError(
            f"Dynamic Hugging Face path must be a file beneath {resource.remote_path.as_posix()!r}: {path_in_repo!r}"
        )
    return remote_path


def _missing_files(directory: Path, filenames: tuple[str, ...]) -> list[str]:
    return [filename for filename in filenames if not (directory / filename).is_file()]


def required_hf_files(category: str, keyword: str) -> tuple[str, ...]:
    """Return the files required for one catalogued directory resource."""
    return _resource(category, keyword).required_files


def download_hf_resource(
    category: str,
    keyword: str,
    *,
    assets_root: str | PathLike[str] | None = None,
    force_download: bool = False,
    path_in_repo: str | None = None,
) -> Path:
    """Download one catalogued project resource without Hugging Face authentication.

    Directory resources mirror their repository paths beneath ``assets_root``.
    When omitted, ``assets_root`` defaults to ``./mmcnirs-assets``. Runtime
    archives are selected dynamically from the downloaded runtime manifest and
    are restricted to paths beneath ``mmc-runtime/``.
    """
    if not isinstance(force_download, bool):
        raise TypeError("force_download must be a boolean")

    resource = _resource(category, keyword)
    root = _resolved_assets_root(assets_root)

    if resource.kind == "directory":
        if path_in_repo is not None:
            raise ValueError("path_in_repo is only supported for dynamic Hugging Face resources")
        snapshot_root = Path(
            snapshot_download(
                repo_id=_REPOSITORY,
                repo_type=_REPO_TYPE,
                revision=_REVISION,
                allow_patterns=f"{resource.remote_path.as_posix()}/**",
                local_dir=root,
                token=False,
                force_download=force_download,
            )
        )
        directory = snapshot_root.joinpath(*resource.remote_path.parts)
        missing = _missing_files(directory, resource.required_files)
        if missing:
            raise FileNotFoundError(
                f"Downloaded {category} resource {keyword!r} is missing: {', '.join(sorted(missing))}"
            )
        return directory

    if resource.kind == "dynamic-file":
        remote_path = _validated_dynamic_path(resource, path_in_repo)
    else:
        if path_in_repo is not None:
            raise ValueError("path_in_repo is only supported for dynamic Hugging Face resources")
        remote_path = resource.remote_path

    downloaded_path = Path(
        hf_hub_download(
            repo_id=_REPOSITORY,
            repo_type=_REPO_TYPE,
            revision=_REVISION,
            filename=remote_path.as_posix(),
            local_dir=root,
            token=False,
            force_download=force_download,
        )
    )
    if not downloaded_path.is_file():
        raise FileNotFoundError(f"Downloaded Hugging Face file does not exist: {downloaded_path}")
    return downloaded_path
