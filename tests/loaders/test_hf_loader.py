from pathlib import Path

import pytest

from mmcnirs.loaders import hf_loader


def _write_required_files(directory: Path, category: str, keyword: str, *, omit: str | None = None) -> None:
    directory.mkdir(parents=True)
    for filename in hf_loader.required_hf_files(category, keyword):
        if filename != omit:
            path = directory / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(filename, encoding="utf-8")


def test_download_directory_uses_default_assets_root_without_authentication(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.chdir(tmp_path)

    def fake_snapshot_download(**kwargs) -> Path:
        calls.append(kwargs)
        _write_required_files(kwargs["local_dir"] / "experiments" / "pain", "experiment", "pain")
        return kwargs["local_dir"]

    monkeypatch.setattr(hf_loader, "snapshot_download", fake_snapshot_download)

    result = hf_loader.download_hf_resource("experiment", "pain")

    assets_root = tmp_path / "mmcnirs-assets"
    assert result == assets_root / "experiments" / "pain"
    assert calls == [
        {
            "repo_id": "nielsbracher/fnirs-simbox-assets",
            "repo_type": "dataset",
            "revision": "main",
            "allow_patterns": "experiments/pain/**",
            "local_dir": assets_root,
            "token": False,
            "force_download": False,
        }
    ]


def test_download_directory_uses_custom_assets_root_and_force_download(tmp_path: Path, monkeypatch) -> None:
    assets_root = tmp_path / "project-data"

    def fake_snapshot_download(**kwargs) -> Path:
        _write_required_files(kwargs["local_dir"] / "e2e-files", "workflow", "e2e-files")
        assert kwargs["force_download"] is True
        return kwargs["local_dir"]

    monkeypatch.setattr(hf_loader, "snapshot_download", fake_snapshot_download)

    assert (
        hf_loader.download_hf_resource(
            "workflow",
            "e2e-files",
            assets_root=assets_root,
            force_download=True,
        )
        == assets_root / "e2e-files"
    )


def test_download_runtime_manifest_and_archive_anonymously(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_hf_hub_download(**kwargs) -> str:
        calls.append(kwargs)
        downloaded_path = kwargs["local_dir"] / kwargs["filename"]
        downloaded_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path.write_text("test", encoding="utf-8")
        return str(downloaded_path)

    monkeypatch.setattr(hf_loader, "hf_hub_download", fake_hf_hub_download)

    manifest = hf_loader.download_hf_resource("runtime", "manifest", assets_root=tmp_path)
    archive = hf_loader.download_hf_resource(
        "runtime",
        "archive",
        assets_root=tmp_path,
        path_in_repo="mmc-runtime/test/runtime.zip",
    )

    assert manifest == tmp_path / "mmc-runtime" / "manifest.json"
    assert archive == tmp_path / "mmc-runtime" / "test" / "runtime.zip"
    assert [call["filename"] for call in calls] == [
        "mmc-runtime/manifest.json",
        "mmc-runtime/test/runtime.zip",
    ]
    assert all(call["token"] is False for call in calls)


@pytest.mark.parametrize(
    "path_in_repo",
    [None, "", "../runtime.zip", "/mmc-runtime/runtime.zip", "standard-heads/runtime.zip", "mmc-runtime"],
)
def test_download_runtime_archive_rejects_missing_or_unsafe_paths(path_in_repo: str | None) -> None:
    with pytest.raises(ValueError, match="path_in_repo|beneath"):
        hf_loader.download_hf_resource("runtime", "archive", path_in_repo=path_in_repo)


def test_download_directory_rejects_incomplete_download(tmp_path: Path, monkeypatch) -> None:
    def fake_snapshot_download(**kwargs) -> Path:
        _write_required_files(
            kwargs["local_dir"] / "standard-heads" / "colin27",
            "standard-head",
            "colin27",
            omit="orientation.txt",
        )
        return kwargs["local_dir"]

    monkeypatch.setattr(hf_loader, "snapshot_download", fake_snapshot_download)

    with pytest.raises(FileNotFoundError, match="orientation.txt"):
        hf_loader.download_hf_resource("standard-head", "colin27", assets_root=tmp_path)


@pytest.mark.parametrize(
    ("category", "keyword", "match"),
    [
        ("unknown", "pain", "category"),
        ("experiment", "unknown", "supported keywords"),
    ],
)
def test_download_rejects_unknown_resources(category: str, keyword: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        hf_loader.download_hf_resource(category, keyword)
