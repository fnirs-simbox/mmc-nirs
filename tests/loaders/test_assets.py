from pathlib import Path

import pytest

from mmc_nirs.loaders import assets


def _write_manifest(experiments_directory: Path, experiment: str) -> None:
    experiment_directory = experiments_directory / experiment
    experiment_directory.mkdir(parents=True)
    (experiment_directory / "assets.yaml").write_text(
        "\n".join(
            [
                "repository: example/assets",
                "repo_type: dataset",
                "revision: test-revision",
                f"remote_path: experiments/{experiment}/assets",
                "files:",
                "  - config.json",
            ]
        ),
        encoding="utf-8",
    )


def test_ensure_experiment_assets_reuses_existing_files(tmp_path: Path, monkeypatch) -> None:
    experiments_directory = tmp_path / "experiments"
    _write_manifest(experiments_directory, "existing")
    assets_directory = experiments_directory / "existing" / "assets"
    assets_directory.mkdir()
    (assets_directory / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(assets, "EXPERIMENTS_DIRECTORY", experiments_directory)

    def unexpected_download(**kwargs) -> None:
        pytest.fail(f"Existing assets should not trigger a download: {kwargs}")

    monkeypatch.setattr(assets, "snapshot_download", unexpected_download)

    assert assets.ensure_experiment_assets("existing") == assets_directory


def test_ensure_experiment_assets_downloads_only_requested_subtree(tmp_path: Path, monkeypatch) -> None:
    experiments_directory = tmp_path / "experiments"
    _write_manifest(experiments_directory, "requested")
    _write_manifest(experiments_directory, "other")
    monkeypatch.setattr(assets, "EXPERIMENTS_DIRECTORY", experiments_directory)
    calls = []

    def fake_download(**kwargs) -> None:
        calls.append(kwargs)
        downloaded_directory = kwargs["local_dir"] / "experiments" / "requested" / "assets"
        downloaded_directory.mkdir(parents=True)
        (downloaded_directory / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(assets, "snapshot_download", fake_download)

    result = assets.ensure_experiment_assets("requested")

    assert result == experiments_directory / "requested" / "assets"
    assert calls == [
        {
            "repo_id": "example/assets",
            "repo_type": "dataset",
            "revision": "test-revision",
            "allow_patterns": "experiments/requested/assets/**",
            "local_dir": tmp_path,
        }
    ]
    assert not (experiments_directory / "other" / "assets").exists()
