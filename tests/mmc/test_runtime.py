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


def _configure_runtime(tmp_path: Path, monkeypatch, system: str, machine: str) -> str:
    token_path = tmp_path / "tokens" / "HF_TOKEN.txt"
    token_path.parent.mkdir()
    token_path.write_text("test-token\n", encoding="utf-8")
    cache_directory = tmp_path / "cache"

    monkeypatch.setattr(runtime, "_TOKEN_PATH", token_path)
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    monkeypatch.setattr(runtime.platform, "machine", lambda: machine)
    monkeypatch.setenv("MMC_NIRS_CACHE_DIR", str(cache_directory))
    return "test-token"


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
    token = _configure_runtime(tmp_path, monkeypatch, system, machine)
    manifest_path, archive_path, remote_archive_path = _make_downloads(tmp_path, platform_key, executable_relative_path)
    calls = []

    def fake_download(**kwargs) -> str:
        calls.append(kwargs)
        return str(manifest_path if kwargs["filename"] == runtime._MANIFEST_PATH else archive_path)

    monkeypatch.setattr(runtime, "hf_hub_download", fake_download)

    executable = runtime.get_mmc_executable()

    assert executable == tmp_path / "cache" / "runtime" / platform_key / executable_relative_path
    assert executable.read_bytes() == b"test executable"
    assert calls == [
        {
            "repo_id": "nielsbracher/fnirs-simbox-assets",
            "repo_type": "dataset",
            "filename": "mmc-runtime/manifest.json",
            "token": token,
        },
        {
            "repo_id": "nielsbracher/fnirs-simbox-assets",
            "repo_type": "dataset",
            "filename": remote_archive_path,
            "token": token,
        },
    ]
    if platform_key != "windows-x86_64":
        assert executable.stat().st_mode & stat.S_IXUSR


def test_get_mmc_executable_reuses_cached_executable_without_download(tmp_path: Path, monkeypatch) -> None:
    _configure_runtime(tmp_path, monkeypatch, "Linux", "x86_64")
    executable = tmp_path / "cache" / "runtime" / "linux-x86_64" / "mmc" / "bin" / "mmc"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"cached")
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def unexpected_download(**kwargs) -> str:
        pytest.fail(f"Cached runtime should not trigger a download: {kwargs}")

    monkeypatch.setattr(runtime, "hf_hub_download", unexpected_download)

    assert runtime.get_mmc_executable() == executable
    assert executable.stat().st_mode & stat.S_IXUSR


def test_get_mmc_executable_rejects_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    _configure_runtime(tmp_path, monkeypatch, "Linux", "x86_64")
    manifest_path, archive_path, _ = _make_downloads(tmp_path, "linux-x86_64", "mmc/bin/mmc")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["linux-x86_64"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_download(**kwargs) -> str:
        return str(manifest_path if kwargs["filename"] == runtime._MANIFEST_PATH else archive_path)

    monkeypatch.setattr(runtime, "hf_hub_download", fake_download)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        runtime.get_mmc_executable()

    assert not (tmp_path / "cache" / "runtime" / "linux-x86_64" / "mmc" / "bin" / "mmc").exists()


def test_get_mmc_executable_rejects_unsupported_platform(tmp_path: Path, monkeypatch) -> None:
    _configure_runtime(tmp_path, monkeypatch, "Linux", "arm64")

    def unexpected_download(**kwargs) -> str:
        pytest.fail(f"Unsupported platform should not trigger a download: {kwargs}")

    monkeypatch.setattr(runtime, "hf_hub_download", unexpected_download)

    with pytest.raises(RuntimeError, match="Unsupported platform.*linux.*arm64"):
        runtime.get_mmc_executable()
