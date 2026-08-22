import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from mmcnirs import load_config, load_light_transport_results

MESH_FIELDS = {
    "nodes",
    "elements",
    "element_tissue_ids",
    "ordered_tissue_ids",
    "ordered_tissues",
}
PROBE_FIELDS = {
    "sourcepos",
    "detpos",
    "sourcedir",
    "detnorms",
    "source_elements",
    "detector_elements",
    "channel_pairings",
    "short_separation_indices",
    "long_separation_indices",
}


@pytest.fixture
def experiment_config_path(tmp_path: Path) -> Path:
    experiment_directory = tmp_path / "pain"
    experiment_directory.mkdir()
    output_directory = experiment_directory / "mmcnirs_outputs"
    output_directory.mkdir()

    np.savez(
        output_directory / "mesh.npz",
        nodes=np.arange(12, dtype=float).reshape(4, 3),
        elements=np.array([[0, 1, 2, 3]]),
        element_tissue_ids=np.array([1]),
        ordered_tissue_ids=np.array([0, 1]),
        ordered_tissues=np.array(["ambient_air", "gray_matter"]),
    )
    np.savez(
        output_directory / "probe.npz",
        sourcepos=np.arange(6, dtype=float).reshape(2, 3),
        detpos=np.arange(9, dtype=float).reshape(3, 3),
        sourcedir=np.ones((2, 3)),
        detnorms=-np.ones((3, 3)),
        source_elements=np.array([0, 0]),
        detector_elements=np.array([0, 0, 0]),
        channel_pairings=np.array([[0, 1], [1, 1], [1, 2]]),
        short_separation_indices=np.array([0]),
        long_separation_indices=np.array([1, 2]),
    )
    for offset, wavelength in enumerate((690, 830)):
        np.savez(
            output_directory / f"jacobian_{wavelength}.npz",
            J=np.arange(24, dtype=float).reshape(6, 4) + offset * 100,
            mea0=np.arange(6, dtype=float).reshape(6, 1) + offset * 10,
            channelidx=np.array([4, 1, 5]),
            Green_s=np.ones((2, 4)),
            Green_d=np.ones((3, 4)),
            Green_sd=np.ones((6, 4)),
        )
    np.savez(
        experiment_directory / "segmentation_map.npz",
        gray_matter=np.array([0, 1, 1, 0]),
        motor_cortex=np.array([False, True, False, False]),
    )

    config_path = experiment_directory / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "Pain",
                "wavelengths": [690, 830],
                "filepaths": {
                    "meshfile": "mmcnirs_outputs/mesh.npz",
                    "jacobians": [
                        "mmcnirs_outputs/jacobian_690.npz",
                        "mmcnirs_outputs/jacobian_830.npz",
                    ],
                    "probefile": "mmcnirs_outputs/probe.npz",
                    "segmentation_map": "segmentation_map.npz",
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_light_transport_results_returns_canonical_inputs(experiment_config_path: Path) -> None:
    data = load_light_transport_results(load_config(experiment_config_path))

    assert set(data) == {"mesh", "probe", "jacobians", "segmentation_map"}
    assert set(data["mesh"]) == MESH_FIELDS
    assert set(data["probe"]) == PROBE_FIELDS
    assert set(data["jacobians"]) == {690, 830}
    assert all(type(wavelength) is int for wavelength in data["jacobians"])

    np.testing.assert_array_equal(data["mesh"]["elements"], [[0, 1, 2, 3]])
    np.testing.assert_array_equal(data["mesh"]["ordered_tissues"], ["ambient_air", "gray_matter"])
    np.testing.assert_array_equal(data["probe"]["channel_pairings"], [[0, 1], [1, 1], [1, 2]])
    np.testing.assert_array_equal(data["probe"]["short_separation_indices"], [0])
    np.testing.assert_array_equal(data["probe"]["long_separation_indices"], [1, 2])

    for offset, wavelength in enumerate((690, 830)):
        result = data["jacobians"][wavelength]
        assert set(result) == {"J", "mea0", "channelidx"}
        np.testing.assert_array_equal(result["J"], np.arange(24).reshape(6, 4) + offset * 100)
        np.testing.assert_array_equal(result["mea0"], np.arange(6).reshape(6, 1) + offset * 10)
        np.testing.assert_array_equal(result["channelidx"], [4, 1, 5])

    assert set(data["segmentation_map"]) == {"gray_matter", "motor_cortex"}
    np.testing.assert_array_equal(data["segmentation_map"]["gray_matter"], [0, 1, 1, 0])
    np.testing.assert_array_equal(data["segmentation_map"]["motor_cortex"], [False, True, False, False])


def test_load_light_transport_results_supports_default_segmentation_path(experiment_config_path: Path) -> None:
    config = load_config(experiment_config_path)
    del config["filepaths"]["segmentation_map"]

    data = load_light_transport_results(config)

    assert set(data["segmentation_map"]) == {"gray_matter", "motor_cortex"}


def test_load_light_transport_results_loads_external_relative_experiment(
    tmp_path: Path,
    experiment_config_path: Path,
) -> None:
    experiment_directory = tmp_path / "experiment_data" / "finger_tapping"
    shutil.copytree(experiment_config_path.parent, experiment_directory)
    config_directory = tmp_path / "configs"
    config_directory.mkdir()

    config = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    config["name"] = "Finger Tapping"
    config["experiment_dir"] = "../experiment_data/finger_tapping"
    config_path = config_directory / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    data = load_light_transport_results(load_config(config_path))

    assert data["mesh"]["nodes"].shape == (4, 3)
    assert set(data["jacobians"]) == {690, 830}


def test_load_light_transport_results_rejects_mismatched_wavelength_and_file_counts(
    experiment_config_path: Path,
) -> None:
    config = load_config(experiment_config_path)
    config["wavelengths"] = [690]

    with pytest.raises(ValueError, match="must have the same length"):
        load_light_transport_results(config)
