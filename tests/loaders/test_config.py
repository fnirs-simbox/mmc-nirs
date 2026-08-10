import pytest

from mmc_nirs import load_config


def test_load_config_returns_bundled_experiment() -> None:
    config = load_config("pain")

    assert config["name"] == "Pain"
    assert config["filepaths"]["meshfile"] == "mesh.npz"


@pytest.mark.parametrize("experiment", ["", "../pain", "pain/config.json"])
def test_load_config_rejects_paths(experiment: str) -> None:
    with pytest.raises(ValueError, match="experiment must be"):
        load_config(experiment)


def test_load_config_reports_unknown_experiment() -> None:
    with pytest.raises(FileNotFoundError, match="unknown"):
        load_config("unknown")
