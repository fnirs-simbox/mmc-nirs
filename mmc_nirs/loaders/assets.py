"""Lazy downloads for bundled experiment assets."""

from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from huggingface_hub import snapshot_download

EXPERIMENTS_DIRECTORY = Path(__file__).parents[1] / "experiments"
_MANIFEST_FIELDS = ("repository", "repo_type", "revision", "remote_path")


def _experiment_directory(experiment: str) -> Path:
    if not experiment or Path(experiment).name != experiment:
        raise ValueError("experiment must be a non-empty name, not a path")
    return EXPERIMENTS_DIRECTORY / experiment


def _load_asset_manifest(experiment: str) -> dict[str, Any]:
    manifest_path = _experiment_directory(experiment) / "assets.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No asset manifest found for experiment {experiment!r}")

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = yaml.safe_load(manifest_file)

    if not isinstance(manifest, dict):
        raise ValueError(f"Asset manifest for experiment {experiment!r} must contain a YAML object")

    for field in _MANIFEST_FIELDS:
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"Asset manifest for experiment {experiment!r} must contain a non-empty {field!r}")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Asset manifest for experiment {experiment!r} must contain a non-empty 'files' list")
    if any(not isinstance(filename, str) or not filename for filename in files):
        raise ValueError(f"Asset manifest for experiment {experiment!r} contains an invalid filename")

    return manifest


def _local_download_root(assets_directory: Path, remote_path: PurePosixPath, experiment: str) -> Path:
    remote_parts = remote_path.parts
    if remote_path.is_absolute() or not remote_parts or any(part in {".", ".."} for part in remote_parts):
        raise ValueError(f"Asset manifest for experiment {experiment!r} contains an invalid 'remote_path'")
    if tuple(assets_directory.parts[-len(remote_parts) :]) != remote_parts:
        raise ValueError(
            f"Remote asset path {remote_path.as_posix()!r} does not map to the local directory for {experiment!r}"
        )
    return assets_directory.parents[len(remote_parts) - 1]


def _missing_files(assets_directory: Path, filenames: list[str], experiment: str) -> list[str]:
    missing: list[str] = []
    for filename in filenames:
        relative_path = PurePosixPath(filename)
        if relative_path.is_absolute() or any(part in {".", ".."} for part in relative_path.parts):
            raise ValueError(f"Asset manifest for experiment {experiment!r} contains an invalid filename")
        if not (assets_directory.joinpath(*relative_path.parts)).is_file():
            missing.append(filename)
    return missing


def ensure_experiment_assets(experiment: str) -> Path:
    """Return the local asset directory, downloading that experiment if needed.

    Existing files listed in the experiment's ``assets.yaml`` are reused. If
    any are missing, only the manifest's remote asset subtree is synchronized
    from Hugging Face. Authentication is discovered by ``huggingface_hub``
    from ``HF_TOKEN`` or credentials saved by ``hf auth login``.
    """
    manifest = _load_asset_manifest(experiment)
    assets_directory = _experiment_directory(experiment) / "assets"
    filenames = manifest["files"]
    missing = _missing_files(assets_directory, filenames, experiment)
    if not missing:
        return assets_directory

    remote_path = PurePosixPath(manifest["remote_path"])
    local_download_root = _local_download_root(assets_directory, remote_path, experiment)
    snapshot_download(
        repo_id=manifest["repository"],
        repo_type=manifest["repo_type"],
        revision=manifest["revision"],
        allow_patterns=f"{remote_path.as_posix()}/**",
        local_dir=local_download_root,
    )

    missing = _missing_files(assets_directory, filenames, experiment)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Downloaded assets for experiment {experiment!r} are missing: {missing_list}")
    return assets_directory
