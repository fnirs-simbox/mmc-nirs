import hashlib
import json
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest

from mmcnirs.loaders import hf_loader
from mmcnirs.mmc import runtime


@pytest.mark.parametrize("system", ["Linux", "Windows"])
@pytest.mark.parametrize(
    ("init_status", "count_status", "device_count", "expected"),
    [(0, 0, 1, True), (0, 0, 0, False), (100, 0, 1, False), (0, 100, 1, False)],
)
def test_cuda_probe_requires_usable_driver_and_device(
    monkeypatch, system, init_status, count_status, device_count, expected
):
    calls = []

    def initialize(flags):
        calls.append(("init", flags))
        return init_status

    def get_count(pointer):
        calls.append(("count",))
        pointer._obj.value = device_count
        return count_status

    class Driver:
        cuInit = staticmethod(initialize)
        cuDeviceGetCount = staticmethod(get_count)

    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    loader = "WinDLL" if system == "Windows" else "CDLL"
    monkeypatch.setattr(runtime.ctypes, loader, lambda name: Driver(), raising=False)
    assert runtime.is_cuda_available() is expected
    assert calls == ([("init", 0)] if init_status else [("init", 0), ("count",)])


def test_cuda_probe_without_driver_or_on_non_cuda_platform(monkeypatch):
    def missing_driver(name):
        raise OSError("driver unavailable")

    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.ctypes, "CDLL", missing_driver)
    assert runtime.is_cuda_available() is False
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    assert runtime.is_cuda_available() is False


@pytest.fixture
def runtime_downloads(tmp_path: Path, monkeypatch):
    """Serve real ZIPs and manifests through the normal HF loader, without network."""
    remote = tmp_path / "remote"
    cache = tmp_path / "cache"
    monkeypatch.setenv("MMCNIRS_CACHE_DIR", str(cache))
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    calls = []
    for version in ("v2025.10", "v2026"):
        artifacts = {}
        for platform_key in ("linux-x86_64", "macos-arm64", "macos-x86_64", "windows-x86_64"):
            folder = "mmc-trinity" if version == "v2026" and not platform_key.startswith("macos") else "mmc"
            filename = "mmc.exe" if platform_key.startswith("windows") else "mmc"
            archive_path = remote / "mmc-runtime" / version / "binaries" / f"{platform_key}.zip"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"{folder}/bin/{filename}", version.encode())
            artifacts[platform_key] = {
                "path": archive_path.relative_to(remote).as_posix(),
                "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            }
        (remote / "mmc-runtime" / version / "manifest.json").write_text(
            json.dumps({"version": version, "artifacts": artifacts}), encoding="utf-8"
        )
    shutil.copyfile(remote / "mmc-runtime/v2026/manifest.json", remote / "mmc-runtime/manifest.json")

    def fake_download(**kwargs):
        calls.append(kwargs)
        destination = kwargs["local_dir"] / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / kwargs["filename"], destination)
        return str(destination)

    monkeypatch.setattr(hf_loader, "hf_hub_download", fake_download)
    return remote, cache, calls


@pytest.mark.parametrize("version", ["v2025.10", "v2026"])
@pytest.mark.parametrize(
    ("system", "machine", "platform_key"),
    [
        ("Linux", "x86_64", "linux-x86_64"),
        ("Darwin", "aarch64", "macos-arm64"),
        ("Darwin", "x86_64", "macos-x86_64"),
        ("Windows", "AMD64", "windows-x86_64"),
    ],
)
def test_fresh_install_uses_root_manifest_version_and_platform_layout(
    runtime_downloads, monkeypatch, version, system, machine, platform_key
) -> None:
    remote, cache, calls = runtime_downloads
    shutil.copyfile(remote / f"mmc-runtime/{version}/manifest.json", remote / "mmc-runtime/manifest.json")
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    monkeypatch.setattr(runtime.platform, "machine", lambda: machine)

    executable = runtime.get_mmc_executable()

    folder = "mmc-trinity" if version == "v2026" and system in {"Linux", "Windows"} else "mmc"
    filename = "mmc.exe" if system == "Windows" else "mmc"
    assert executable == cache / "runtime" / version / platform_key / folder / "bin" / filename
    assert executable.read_bytes() == version.encode()
    assert [call["filename"] for call in calls] == [
        "mmc-runtime/manifest.json",
        f"mmc-runtime/{version}/binaries/{platform_key}.zip",
    ]
    if os.name != "nt" and system != "Windows":
        assert executable.stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize("version", ["v2025.10", "v2026"])
def test_explicit_version_uses_its_manifest_and_reuses_installation(runtime_downloads, version) -> None:
    _, _, calls = runtime_downloads

    executable = runtime.get_mmc_executable(version=version)

    assert executable.read_bytes() == version.encode()
    assert [call["filename"] for call in calls] == [
        f"mmc-runtime/{version}/manifest.json",
        f"mmc-runtime/{version}/binaries/linux-x86_64.zip",
    ]
    calls.clear()
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert runtime.get_mmc_executable(version=version) == executable
    assert calls == []
    if os.name != "nt":
        assert executable.stat().st_mode & stat.S_IXUSR


def test_default_update_installs_changed_version_and_preserves_both_versions(runtime_downloads) -> None:
    remote, _, calls = runtime_downloads
    shutil.copyfile(remote / "mmc-runtime/v2025.10/manifest.json", remote / "mmc-runtime/manifest.json")
    old_executable = runtime.get_mmc_executable()
    shutil.copyfile(remote / "mmc-runtime/v2026/manifest.json", remote / "mmc-runtime/manifest.json")
    calls.clear()

    assert runtime.get_mmc_executable(update_manifest=False) == old_executable
    assert calls == []

    new_executable = runtime.get_mmc_executable(update_manifest=True)

    assert new_executable != old_executable
    assert old_executable.read_bytes() == b"v2025.10"
    assert new_executable.read_bytes() == b"v2026"
    assert [call["filename"] for call in calls] == [
        "mmc-runtime/manifest.json",
        "mmc-runtime/v2026/binaries/linux-x86_64.zip",
    ]
    assert calls[0]["force_download"] is True
    calls.clear()
    assert runtime.get_mmc_executable() == new_executable
    assert calls == []
    assert runtime.get_mmc_executable(version="v2025.10") == old_executable
    assert runtime.get_mmc_executable(version="v2026") == new_executable
    # Explicit manifests can be fetched, but neither archive is downloaded again.
    assert all(call["filename"].endswith("manifest.json") for call in calls)


def test_update_with_unchanged_default_only_fetches_root_manifest(runtime_downloads) -> None:
    _, _, calls = runtime_downloads
    executable = runtime.get_mmc_executable()
    calls.clear()

    assert runtime.get_mmc_executable(update_manifest=True) == executable
    assert len(calls) == 1
    assert calls[0]["filename"] == "mmc-runtime/manifest.json"
    assert calls[0]["force_download"] is True


def test_explicit_version_takes_precedence_when_updating_default(runtime_downloads) -> None:
    _, _, calls = runtime_downloads
    old_executable = runtime.get_mmc_executable(version="v2025.10")
    calls.clear()

    assert runtime.get_mmc_executable(version="v2025.10", update_manifest=True) == old_executable
    assert len(calls) == 1
    assert calls[0]["filename"] == "mmc-runtime/manifest.json"
    assert calls[0]["force_download"] is True
    assert runtime.get_mmc_executable().read_bytes() == b"v2026"


def test_get_mmc_executable_rejects_hash_mismatch(runtime_downloads) -> None:
    remote, cache, _ = runtime_downloads
    manifest_path = remote / "mmc-runtime/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["linux-x86_64"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        runtime.get_mmc_executable()

    assert not (cache / "runtime").exists()


def test_explicit_manifest_must_declare_requested_version(runtime_downloads) -> None:
    remote, _, _ = runtime_downloads
    shutil.copyfile(remote / "mmc-runtime/v2026/manifest.json", remote / "mmc-runtime/v2025.10/manifest.json")

    with pytest.raises(RuntimeError, match="declares 'v2026', expected 'v2025.10'"):
        runtime.get_mmc_executable(version="v2025.10")


@pytest.mark.parametrize("version", ["", "..", "../v2026", "/v2026", "v2026\\file"])
def test_invalid_version_is_rejected_without_download(runtime_downloads, version) -> None:
    _, _, calls = runtime_downloads
    with pytest.raises(ValueError, match="Invalid MMC runtime version"):
        runtime.get_mmc_executable(version=version)
    assert calls == []


def test_get_mmc_executable_rejects_unsupported_platform(runtime_downloads, monkeypatch) -> None:
    _, _, calls = runtime_downloads
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")

    with pytest.raises(RuntimeError, match="Unsupported platform.*linux.*arm64"):
        runtime.get_mmc_executable()
    assert calls == []


@pytest.mark.skipif(
    os.environ.get("MMCNIRS_TEST_RUNTIME_DOWNLOADS") != "1",
    reason="Set MMCNIRS_TEST_RUNTIME_DOWNLOADS=1 to download and launch the real v2026 runtime",
)
def test_real_v2026_executable_launches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MMCNIRS_CACHE_DIR", str(tmp_path / "cache"))
    executable = runtime.get_mmc_executable(version="v2026")

    result = subprocess.run([str(executable), "--version"], capture_output=True, text=True, timeout=30, check=True)

    assert "2026" in result.stdout + result.stderr
