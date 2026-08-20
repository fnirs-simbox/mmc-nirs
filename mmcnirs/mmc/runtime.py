"""Discovery and managed installation of the external MMC executable."""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from mmcnirs.loaders.hf_loader import download_hf_resource

__all__ = ["get_mmc_executable"]

_EXECUTABLE_PATHS = {
    "linux-x86_64": PurePosixPath("mmc/bin/mmc"),
    "macos-arm64": PurePosixPath("mmc/bin/mmc"),
    "macos-x86_64": PurePosixPath("mmc/bin/mmc"),
    "windows-x86_64": PurePosixPath("mmc/bin/mmc.exe"),
}


def _detect_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower().replace("-", "_")
    if machine in {"amd64", "x64"}:
        machine = "x86_64"
    elif machine == "aarch64":
        machine = "arm64"

    platform_key = {
        ("linux", "x86_64"): "linux-x86_64",
        ("darwin", "arm64"): "macos-arm64",
        ("darwin", "x86_64"): "macos-x86_64",
        ("windows", "x86_64"): "windows-x86_64",
    }.get((system, machine))
    if platform_key is None:
        supported = ", ".join(_EXECUTABLE_PATHS)
        raise RuntimeError(
            f"Unsupported platform for the MMC runtime: system={system!r}, machine={machine!r}. "
            f"Supported platforms: {supported}."
        )
    return platform_key


def _user_cache_directory(platform_key: str) -> Path:
    override = os.environ.get("MMCNIRS_CACHE_DIR")
    if override:
        return Path(override).expanduser()

    if platform_key.startswith("windows-"):
        cache_base = os.environ.get("LOCALAPPDATA")
        if cache_base:
            return Path(cache_base) / "mmcnirs" / "Cache"
        return Path.home() / "AppData" / "Local" / "mmcnirs" / "Cache"
    if platform_key.startswith("macos-"):
        return Path.home() / "Library" / "Caches" / "mmcnirs"

    cache_base = os.environ.get("XDG_CACHE_HOME")
    return (Path(cache_base).expanduser() if cache_base else Path.home() / ".cache") / "mmcnirs"


def _executable_path(cache_directory: Path, platform_key: str) -> Path:
    return cache_directory.joinpath("runtime", platform_key, *_EXECUTABLE_PATHS[platform_key].parts)


def _ensure_executable_permission(executable: Path, platform_key: str) -> None:
    if not platform_key.startswith("windows-"):
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _load_artifact(manifest_path: Path, platform_key: str) -> tuple[str, str]:
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest: Any = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read MMC runtime manifest at {manifest_path}: {error}") from error

    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    artifact = artifacts.get(platform_key) if isinstance(artifacts, dict) else None
    if not isinstance(artifact, dict):
        raise RuntimeError(f"MMC runtime manifest has no artifact for {platform_key!r}")

    archive_path = artifact.get("path")
    expected_sha256 = artifact.get("sha256")
    if not isinstance(archive_path, str) or not archive_path:
        raise RuntimeError(f"MMC runtime manifest contains an invalid path for {platform_key!r}")
    remote_path = PurePosixPath(archive_path)
    if remote_path.is_absolute() or any(part in {".", ".."} for part in remote_path.parts):
        raise RuntimeError(f"MMC runtime manifest contains an unsafe path for {platform_key!r}: {archive_path!r}")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256)
    ):
        raise RuntimeError(f"MMC runtime manifest contains an invalid SHA-256 for {platform_key!r}")
    return remote_path.as_posix(), expected_sha256.lower()


def _verify_archive(archive_path: Path, expected_sha256: str, platform_key: str) -> None:
    try:
        with archive_path.open("rb") as archive_file:
            actual_sha256 = hashlib.file_digest(archive_file, "sha256").hexdigest()
    except OSError as error:
        raise RuntimeError(f"Could not read downloaded MMC runtime archive at {archive_path}: {error}") from error

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"MMC runtime archive SHA-256 mismatch for {platform_key}: expected {expected_sha256}, got {actual_sha256}."
        )


def _extract_zip(archive_path: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = PurePosixPath(member.filename)
                if (
                    member_path.is_absolute()
                    or any(part in {".", ".."} for part in member_path.parts)
                    or "\\" in member.filename
                ):
                    raise RuntimeError(f"MMC runtime archive contains an unsafe path: {member.filename!r}")
            archive.extractall(destination)
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"Could not extract MMC runtime archive at {archive_path}: {error}") from error


def _install_archive(archive_path: Path, cache_directory: Path, platform_key: str) -> Path:
    install_directory = cache_directory / "runtime" / platform_key
    executable_relative_path = _EXECUTABLE_PATHS[platform_key]
    install_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(prefix=f".{platform_key}-", dir=install_directory.parent))

    try:
        _extract_zip(archive_path, temporary_directory)
        temporary_executable = temporary_directory.joinpath(*executable_relative_path.parts)
        if not temporary_executable.is_file():
            raise RuntimeError(
                f"MMC runtime archive for {platform_key} does not contain {executable_relative_path.as_posix()!r}"
            )
        _ensure_executable_permission(temporary_executable, platform_key)

        if install_directory.exists():
            cached_executable = install_directory.joinpath(*executable_relative_path.parts)
            if cached_executable.is_file():
                _ensure_executable_permission(cached_executable, platform_key)
                return cached_executable
            shutil.rmtree(install_directory)

        try:
            temporary_directory.replace(install_directory)
        except OSError:
            # Another process may have completed the same atomic installation.
            cached_executable = install_directory.joinpath(*executable_relative_path.parts)
            if not cached_executable.is_file():
                raise
            _ensure_executable_permission(cached_executable, platform_key)
            return cached_executable

        executable = install_directory.joinpath(*executable_relative_path.parts)
        _ensure_executable_permission(executable, platform_key)
        return executable
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)


def get_mmc_executable() -> Path:
    """Return the platform's MMC executable, downloading and caching it if needed.

    The archive is selected from the public runtime manifest, downloaded
    anonymously to the current user's managed cache, verified before
    extraction, and installed in that cache.
    """
    platform_key = _detect_platform()
    cache_directory = _user_cache_directory(platform_key)
    cached_executable = _executable_path(cache_directory, platform_key)
    if cached_executable.is_file():
        _ensure_executable_permission(cached_executable, platform_key)
        return cached_executable

    manifest_path = download_hf_resource("runtime", "manifest", assets_root=cache_directory)
    archive_remote_path, expected_sha256 = _load_artifact(manifest_path, platform_key)
    archive_path = download_hf_resource(
        "runtime",
        "archive",
        assets_root=cache_directory,
        path_in_repo=archive_remote_path,
    )
    _verify_archive(archive_path, expected_sha256, platform_key)
    return _install_archive(archive_path, cache_directory, platform_key)
