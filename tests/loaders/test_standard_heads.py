from pathlib import Path

import pytest

from mmcnirs.loaders import standard_heads

_FILES = ("README.md", "colin27_mesh.npz", "orientation.txt", "segmentation_map.npz")


def _write_head(directory: Path, *, omit: str | None = None) -> None:
    directory.mkdir(parents=True)
    for filename in _FILES:
        if filename != omit:
            (directory / filename).write_text(filename, encoding="utf-8")


def test_load_standard_head_reuses_complete_saved_directory(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "colin27"
    _write_head(destination)

    def unexpected_download(*args, **kwargs) -> None:
        pytest.fail(f"A complete saved head should not trigger a download: {args}, {kwargs}")

    monkeypatch.setattr(standard_heads, "download_hf_resource", unexpected_download)

    assert standard_heads.load_standard_head("colin27", save=True, directory=tmp_path) == destination


def test_load_standard_head_downloads_and_saves_requested_subtree(tmp_path: Path, monkeypatch) -> None:
    assets_root = tmp_path / "mmcnirs-assets"
    downloaded_directory = assets_root / "standard-heads" / "colin27"
    _write_head(downloaded_directory)
    calls = []

    def fake_download(*args, **kwargs) -> Path:
        calls.append((args, kwargs))
        return downloaded_directory

    monkeypatch.setattr(standard_heads, "download_hf_resource", fake_download)

    result = standard_heads.load_standard_head(
        "colin27",
        save=True,
        directory=tmp_path,
        assets_root=assets_root,
        overwrite=True,
    )

    assert result == tmp_path / "colin27"
    assert sorted(path.name for path in result.iterdir()) == sorted(_FILES)
    assert calls == [(("standard-head", "colin27"), {"assets_root": assets_root, "force_download": True})]


def test_load_standard_head_returns_central_download_directory_without_copying(tmp_path: Path, monkeypatch) -> None:
    downloaded_directory = tmp_path / "mmcnirs-assets" / "standard-heads" / "colin27"
    _write_head(downloaded_directory)
    monkeypatch.setattr(standard_heads, "download_hf_resource", lambda *args, **kwargs: downloaded_directory)

    assert standard_heads.load_standard_head("colin27") == downloaded_directory


def test_load_standard_head_rejects_incomplete_download(tmp_path: Path, monkeypatch) -> None:
    downloaded_directory = tmp_path / "mmcnirs-assets" / "standard-heads" / "colin27"
    _write_head(downloaded_directory, omit="orientation.txt")
    monkeypatch.setattr(standard_heads, "download_hf_resource", lambda *args, **kwargs: downloaded_directory)

    with pytest.raises(FileNotFoundError, match="orientation.txt"):
        standard_heads.load_standard_head("colin27")


@pytest.mark.parametrize("name", ["", "../colin27", "standard-heads/colin27", "unknown"])
def test_load_standard_head_rejects_invalid_or_unknown_names(name: str) -> None:
    with pytest.raises(ValueError, match="standard head|standard_head"):
        standard_heads.load_standard_head(name)
