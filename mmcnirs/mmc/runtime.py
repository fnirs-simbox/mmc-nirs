"""Discovery and managed installation of the external MMC executable."""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from mmcnirs.loaders.hf_loader import download_hf_resource

__all__ = ["get_mmc_executable", "is_cuda_available"]


def is_cuda_available() -> bool:
    """Return whether the NVIDIA driver can initialize and enumerate a device.

    This probe allocates no GPU context and needs neither PyTorch nor a CUDA
    toolkit. It does not install MMC or test a particular MMC binary. The
    workflow uses it to offer CUDA with the CUDA-capable managed runtime;
    OpenCL and CPU remain explicit alternatives.
    """
    system = platform.system()
    if system not in {"Linux", "Windows"}:
        return False
    try:
        driver = ctypes.WinDLL("nvcuda.dll") if system == "Windows" else ctypes.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        if driver.cuInit(0) != 0:
            return False
        count = ctypes.c_int()
        return driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except (OSError, AttributeError):
        return False


_EXECUTABLE_PATHS = {
    "linux-x86_64": ("mmc/bin/mmc", "mmc-trinity/bin/mmc"),
    "macos-arm64": ("mmc/bin/mmc",),
    "macos-x86_64": ("mmc/bin/mmc",),
    "windows-x86_64": ("mmc/bin/mmc.exe", "mmc-trinity/bin/mmc.exe"),
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


def _find_executable(directory: Path, platform_key: str) -> Path | None:
    for relative_path in _EXECUTABLE_PATHS[platform_key]:
        executable = directory.joinpath(*PurePosixPath(relative_path).parts)
        if executable.is_file():
            return executable
    return None


def _ensure_executable_permission(executable: Path, platform_key: str) -> None:
    if not platform_key.startswith("windows-"):
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _validate_version(version: str) -> str:
    if not isinstance(version, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version) is None:
        raise ValueError(f"Invalid MMC runtime version: {version!r}")
    return version


def _load_manifest(cache_directory: Path, version: str | None = None, *, refresh: bool = False) -> dict[str, Any]:
    remote_path = f"mmc-runtime/{version}/manifest.json" if version else "mmc-runtime/manifest.json"
    manifest_path = cache_directory / remote_path
    if refresh or not manifest_path.is_file():
        manifest_path = download_hf_resource(
            "runtime",
            "manifest",
            assets_root=cache_directory,
            path_in_repo=remote_path,
            force_download=refresh,
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest: Any = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read MMC runtime manifest at {manifest_path}: {error}") from error

    try:
        declared_version = _validate_version(manifest.get("version") if isinstance(manifest, dict) else None)
    except ValueError as error:
        raise RuntimeError(f"MMC runtime manifest at {manifest_path} contains an invalid version") from error
    if version is not None and version != declared_version:
        raise RuntimeError(f"MMC runtime manifest declares {declared_version!r}, expected {version!r}")
    return manifest


def _load_artifact(manifest: dict[str, Any], platform_key: str) -> tuple[str, str]:
    artifacts = manifest.get("artifacts")
    artifact = artifacts.get(platform_key) if isinstance(artifacts, dict) else None
    if not isinstance(artifact, dict):
        raise RuntimeError(f"MMC runtime manifest has no artifact for {platform_key!r}")

    archive_path = artifact.get("path")
    expected_sha256 = artifact.get("sha256")
    if not isinstance(archive_path, str) or not archive_path:
        raise RuntimeError(f"MMC runtime manifest contains an invalid path for {platform_key!r}")
    remote_path = PurePosixPath(archive_path)
    if (
        remote_path.is_absolute()
        or any(part in {".", ".."} for part in remote_path.parts)
        or "\\" in archive_path
        or not remote_path.is_relative_to("mmc-runtime")
        or remote_path == PurePosixPath("mmc-runtime")
    ):
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


def _install_archive(archive_path: Path, install_directory: Path, platform_key: str) -> Path:
    install_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(prefix=f".{platform_key}-", dir=install_directory.parent))

    try:
        _extract_zip(archive_path, temporary_directory)
        temporary_executable = _find_executable(temporary_directory, platform_key)
        if temporary_executable is None:
            raise RuntimeError(
                f"MMC runtime archive for {platform_key} does not contain any of {_EXECUTABLE_PATHS[platform_key]}"
            )
        _ensure_executable_permission(temporary_executable, platform_key)

        if install_directory.exists():
            cached_executable = _find_executable(install_directory, platform_key)
            if cached_executable is not None:
                _ensure_executable_permission(cached_executable, platform_key)
                return cached_executable
            shutil.rmtree(install_directory)

        try:
            temporary_directory.replace(install_directory)
        except OSError:
            # Another process may have completed the same atomic installation.
            cached_executable = _find_executable(install_directory, platform_key)
            if cached_executable is None:
                raise
            _ensure_executable_permission(cached_executable, platform_key)
            return cached_executable

        executable = install_directory / temporary_executable.relative_to(temporary_directory)
        _ensure_executable_permission(executable, platform_key)
        return executable
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)


def get_mmc_executable(version: str | None = None, update_manifest: bool = False) -> Path:
    """Return the platform's MMC executable, downloading and caching it if needed.

    With no explicit ``version``, the cached root manifest selects the default.
    ``update_manifest=True`` refreshes that manifest from Hugging Face so a
    changed default is installed and used. An explicit version always takes
    precedence and uses its own manifest. Without an update request, cached
    manifests and installations are reused without remote checks.

    Archives are SHA-256 verified before extraction. Versions coexist beneath
    ``runtime/<version>/<platform>`` in the user's managed cache.
    """
    if version is not None:
        _validate_version(version)
    if not isinstance(update_manifest, bool):
        raise TypeError("update_manifest must be a boolean")
    platform_key = _detect_platform()
    cache_directory = _user_cache_directory(platform_key)

    if version is None:
        manifest = _load_manifest(cache_directory, refresh=update_manifest)
    else:
        if update_manifest:
            _load_manifest(cache_directory, refresh=True)
        manifest = _load_manifest(cache_directory, version)
    install_directory = cache_directory / "runtime" / manifest["version"] / platform_key
    cached_executable = _find_executable(install_directory, platform_key)
    if cached_executable is not None:
        _ensure_executable_permission(cached_executable, platform_key)
        return cached_executable

    archive_remote_path, expected_sha256 = _load_artifact(manifest, platform_key)
    archive_path = cache_directory / archive_remote_path
    if not archive_path.is_file():
        archive_path = download_hf_resource(
            "runtime",
            "archive",
            assets_root=cache_directory,
            path_in_repo=archive_remote_path,
        )
    _verify_archive(archive_path, expected_sha256, platform_key)
    return _install_archive(archive_path, install_directory, platform_key)
