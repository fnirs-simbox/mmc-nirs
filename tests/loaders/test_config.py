import json

import pytest

from mmc_nirs import load_config, load_default_config


def test_load_config_returns_bundled_experiment(tmp_path, monkeypatch) -> None:
    assets_directory = tmp_path / "pain" / "assets"
    assets_directory.mkdir(parents=True)
    (assets_directory / "config.json").write_text(
        '{"name": "Pain", "experiment_dir": "ignored", "filepaths": {"meshfile": "mesh.npz"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr("mmc_nirs.loaders.config.ensure_experiment_assets", lambda experiment: assets_directory)

    config = load_default_config("pain")

    assert config["name"] == "Pain"
    assert config["filepaths"]["meshfile"] == "mesh.npz"
    assert config["experiment_dir"] == assets_directory


@pytest.mark.parametrize("experiment", ["", "../pain", "pain/config.json"])
def test_load_config_rejects_paths(experiment: str) -> None:
    with pytest.raises(ValueError, match="experiment must be"):
        load_default_config(experiment)


def test_load_config_reports_unknown_experiment() -> None:
    with pytest.raises(FileNotFoundError, match="unknown"):
        load_default_config("unknown")


def test_load_config_defaults_to_config_directory(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"filepaths": {}}', encoding="utf-8")

    config = load_config(config_path)

    assert config["experiment_dir"] == tmp_path


def test_load_config_resolves_user_output_directory_without_requiring_it_to_exist(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"experiment_dir": "generated/output", "filepaths": {}}', encoding="utf-8")

    config = load_config(config_path)

    assert config["experiment_dir"] == tmp_path / "generated" / "output"
    assert not config["experiment_dir"].exists()


def test_load_config_preserves_absolute_user_output_directory(tmp_path) -> None:
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    output_dir = tmp_path / "generated" / "output"
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps({"experiment_dir": str(output_dir), "filepaths": {}}), encoding="utf-8")

    config = load_config(config_path)

    assert config["experiment_dir"] == output_dir
