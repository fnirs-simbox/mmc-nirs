from pathlib import Path

import pytest

from mmc_nirs.loaders import standard_heads

_FILES = ("README.md", "colin27_mesh.npz", "orientation.txt", "segmentation_map.npz")


def _write_head(directory: Path, *, omit: str | None = None) -> None:
    directory.mkdir(parents=True)
    for filename in _FILES:
        if filename != omit:
            (directory / filename).write_text(filename, encoding="utf-8")


def test_load_standard_head_reuses_complete_saved_directory(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "colin27"
    _write_head(destination)

    def unexpected_download(**kwargs) -> None:
        pytest.fail(f"A complete saved head should not trigger a download: {kwargs}")

    monkeypatch.setattr(standard_heads, "snapshot_download", unexpected_download)

    assert standard_heads.load_standard_head("colin27", save=True, directory=tmp_path) == destination


def test_load_standard_head_downloads_and_saves_requested_subtree(tmp_path: Path, monkeypatch) -> None:
    token_path = tmp_path / "HF_TOKEN.txt"
    token_path.write_text("test-token", encoding="utf-8")
    snapshot_root = tmp_path / "cache" / "snapshot"
    cached_directory = snapshot_root / "standard_heads" / "colin27"
    calls = []

    def fake_download(**kwargs) -> Path:
        calls.append(kwargs)
        _write_head(cached_directory)
        return snapshot_root

    monkeypatch.setattr(standard_heads, "_TOKEN_PATH", token_path)
    monkeypatch.setattr(standard_heads, "snapshot_download", fake_download)

    result = standard_heads.load_standard_head("colin27", save=True, directory=tmp_path, overwrite=True)

    assert result == tmp_path / "colin27"
    assert sorted(path.name for path in result.iterdir()) == sorted(_FILES)
    assert calls == [
        {
            "repo_id": "nielsbracher/fnirs-simbox-assets",
            "repo_type": "dataset",
            "allow_patterns": "standard_heads/colin27/**",
            "token": "test-token",
            "force_download": True,
        }
    ]


def test_load_standard_head_can_return_cached_directory_without_copying(tmp_path: Path, monkeypatch) -> None:
    token_path = tmp_path / "HF_TOKEN.txt"
    token_path.write_text("test-token", encoding="utf-8")
    snapshot_root = tmp_path / "cache" / "snapshot"
    cached_directory = snapshot_root / "standard_heads" / "colin27"
    _write_head(cached_directory)

    monkeypatch.setattr(standard_heads, "_TOKEN_PATH", token_path)
    monkeypatch.setattr(standard_heads, "snapshot_download", lambda **kwargs: snapshot_root)

    assert standard_heads.load_standard_head("colin27") == cached_directory


def test_load_standard_head_rejects_incomplete_download(tmp_path: Path, monkeypatch) -> None:
    token_path = tmp_path / "HF_TOKEN.txt"
    token_path.write_text("test-token", encoding="utf-8")
    snapshot_root = tmp_path / "cache" / "snapshot"
    _write_head(snapshot_root / "standard_heads" / "colin27", omit="orientation.txt")

    monkeypatch.setattr(standard_heads, "_TOKEN_PATH", token_path)
    monkeypatch.setattr(standard_heads, "snapshot_download", lambda **kwargs: snapshot_root)

    with pytest.raises(FileNotFoundError, match="orientation.txt"):
        standard_heads.load_standard_head("colin27")


@pytest.mark.parametrize("name", ["", "../colin27", "standard_heads/colin27", "unknown"])
def test_load_standard_head_rejects_invalid_or_unknown_names(name: str) -> None:
    with pytest.raises(ValueError, match="standard head|standard_head"):
        standard_heads.load_standard_head(name)


def test_load_standard_head_requires_private_dataset_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(standard_heads, "_TOKEN_PATH", tmp_path / "missing-token.txt")

    with pytest.raises(RuntimeError, match="token file not found"):
        standard_heads.load_standard_head("colin27")
