import pytest

from mmc_nirs import load_config, load_default_config


def test_load_config_returns_bundled_experiment(tmp_path, monkeypatch) -> None:
    assets_directory = tmp_path / "pain" / "assets"
    assets_directory.mkdir(parents=True)
    (assets_directory / "config.json").write_text(
        '{"name": "Pain", "filepaths": {"meshfile": "mesh.npz"}}', encoding="utf-8"
    )
    monkeypatch.setattr("mmc_nirs.loaders.config.ensure_experiment_assets", lambda experiment: assets_directory)

    config = load_default_config("pain")

    assert config["name"] == "Pain"
    assert config["filepaths"]["meshfile"] == "mesh.npz"
    assert config["filepaths"]["experiment_directory"] == assets_directory


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

    assert config["filepaths"]["experiment_directory"] == tmp_path
