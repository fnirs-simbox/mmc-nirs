import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest

from mmc_nirs.mmc import runtime


def _make_downloads(tmp_path: Path, platform_key: str, executable_path: str) -> tuple[Path, Path, str]:
    archive_path = tmp_path / f"{platform_key}.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(executable_path, b"test executable")
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    remote_archive_path = f"mmc-runtime/test/{platform_key}.zip"

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "test",
                "artifacts": {
                    platform_key: {
                        "path": remote_archive_path,
                        "sha256": archive_sha256,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, archive_path, remote_archive_path


def _configure_runtime(tmp_path: Path, monkeypatch, system: str, machine: str) -> None:
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    monkeypatch.setattr(runtime.platform, "machine", lambda: machine)
    monkeypatch.setenv("MMC_NIRS_CACHE_DIR", str(tmp_path / "cache"))


@pytest.mark.parametrize(
    ("system", "machine", "platform_key", "executable_relative_path"),
    [
        ("Linux", "x86_64", "linux-x86_64", "mmc/bin/mmc"),
        ("Darwin", "arm64", "macos-arm64", "mmc/bin/mmc"),
        ("Darwin", "AMD64", "macos-x86_64", "mmc/bin/mmc"),
        ("Windows", "AMD64", "windows-x86_64", "mmc/bin/mmc.exe"),
    ],
)
def test_get_mmc_executable_downloads_matching_verified_archive(
    tmp_path: Path,
    monkeypatch,
    system: str,
    machine: str,
    platform_key: str,
    executable_relative_path: str,
) -> None:
    _configure_runtime(tmp_path, monkeypatch, system, machine)
    manifest_path, archive_path, remote_archive_path = _make_downloads(tmp_path, platform_key, executable_relative_path)
    cache_directory = tmp_path / "cache"
    calls = []

    def fake_download(*args, **kwargs) -> Path:
        calls.append((args, kwargs))
        return manifest_path if args == ("runtime", "manifest") else archive_path

    monkeypatch.setattr(runtime, "download_hf_resource", fake_download)

    executable = runtime.get_mmc_executable()

    assert executable == tmp_path / "cache" / "runtime" / platform_key / executable_relative_path
    assert executable.read_bytes() == b"test executable"
    assert calls == [
        (("runtime", "manifest"), {"assets_root": cache_directory}),
        (
            ("runtime", "archive"),
            {
                "assets_root": cache_directory,
                "path_in_repo": remote_archive_path,
            },
        ),
    ]
    if platform_key != "windows-x86_64":
        assert executable.stat().st_mode & stat.S_IXUSR


def test_get_mmc_executable_reuses_cached_executable_without_download(tmp_path: Path, monkeypatch) -> None:
    _configure_runtime(tmp_path, monkeypatch, "Linux", "x86_64")
    executable = tmp_path / "cache" / "runtime" / "linux-x86_64" / "mmc" / "bin" / "mmc"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"cached")
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def unexpected_download(*args, **kwargs) -> Path:
        pytest.fail(f"Cached runtime should not trigger a download: {args}, {kwargs}")

    monkeypatch.setattr(runtime, "download_hf_resource", unexpected_download)

    assert runtime.get_mmc_executable() == executable
    assert executable.stat().st_mode & stat.S_IXUSR


def test_get_mmc_executable_rejects_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    _configure_runtime(tmp_path, monkeypatch, "Linux", "x86_64")
    manifest_path, archive_path, _ = _make_downloads(tmp_path, "linux-x86_64", "mmc/bin/mmc")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["linux-x86_64"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_download(*args, **kwargs) -> Path:
        return manifest_path if args == ("runtime", "manifest") else archive_path

    monkeypatch.setattr(runtime, "download_hf_resource", fake_download)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        runtime.get_mmc_executable()

    assert not (tmp_path / "cache" / "runtime" / "linux-x86_64" / "mmc" / "bin" / "mmc").exists()


def test_get_mmc_executable_rejects_unsupported_platform(tmp_path: Path, monkeypatch) -> None:
    _configure_runtime(tmp_path, monkeypatch, "Linux", "arm64")

    def unexpected_download(*args, **kwargs) -> Path:
        pytest.fail(f"Unsupported platform should not trigger a download: {args}, {kwargs}")

    monkeypatch.setattr(runtime, "download_hf_resource", unexpected_download)

    with pytest.raises(RuntimeError, match="Unsupported platform.*linux.*arm64"):
        runtime.get_mmc_executable()
