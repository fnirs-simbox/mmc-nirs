import json

import numpy as np
import pytest

from mmcnirs.utils.jacobian_utils import (
    build_jacobian_mmc_config,
    mmc_to_json,
    order_optical_properties,
    save_mmc_mesh,
    select_optical_properties,
    validate_jacobian,
    validate_mmc_flux,
    validate_mmc_settings,
)


def test_validate_jacobian_accepts_full_source_detector_matrix() -> None:
    jacobian = np.arange(24, dtype=float).reshape(4, 6)

    validated = validate_jacobian(jacobian, source_count=2, detector_count=2, node_count=6)

    np.testing.assert_array_equal(validated, jacobian)


@pytest.mark.parametrize(
    ("jacobian", "error", "message"),
    [
        (np.ones((3, 6)), ValueError, "shape"),
        (np.full((4, 6), np.nan), ValueError, "non-finite"),
        (np.full((4, 6), "invalid"), TypeError, "real numeric"),
        (np.ones((4, 6), dtype=complex), TypeError, "real numeric"),
    ],
)
def test_validate_jacobian_rejects_invalid_values(jacobian, error, message) -> None:
    with pytest.raises(error, match=message):
        validate_jacobian(jacobian, source_count=2, detector_count=2, node_count=6)


def test_order_optical_properties_uses_configured_tissue_order() -> None:
    properties = {
        "690": {
            "scalp": [0.02, 0.8, 0.9, 1.37],
            "ambient_air": [0.0, 0.0, 1.0, 1.0],
            "brain": [0.03, 7.0, 0.9, 1.37],
        }
    }

    ordered = order_optical_properties(properties, {"2": "scalp", "0": "ambient_air", "1": "brain"})

    assert ordered == {
        "690": [
            [0.0, 0.0, 1.0, 1.0],
            [0.03, 7.0, 0.9, 1.37],
            [0.02, 0.8, 0.9, 1.37],
        ]
    }


def test_order_optical_properties_rejects_missing_tissue() -> None:
    with pytest.raises(ValueError, match="missing tissue 'brain'"):
        order_optical_properties(
            {"690": {"ambient_air": [0.0, 0.0, 1.0, 1.0]}},
            {"0": "ambient_air", "1": "brain"},
        )


def test_select_optical_properties_returns_one_ordered_wavelength() -> None:
    selected = select_optical_properties(
        {"690": {"air": [0.0, 0.0, 1.0, 1.0], "tissue": [0.1, 1.0, 0.9, 1.4]}},
        {"1": "tissue", "0": "air"},
        690,
    )

    np.testing.assert_array_equal(selected, [[0.0, 0.0, 1.0, 1.0], [0.1, 1.0, 0.9, 1.4]])


def test_validate_mmc_settings_accepts_integer_valued_photon_count() -> None:
    assert validate_mmc_settings({"nphoton": 5e9}) == 5_000_000_000

    with pytest.raises(ValueError, match="unsupported field"):
        validate_mmc_settings({"nphoton": 100, "tstep": 1.0})


def test_build_jacobian_mmc_config_converts_elements_to_one_based() -> None:
    config = build_jacobian_mmc_config(
        np.zeros((4, 3)),
        [[0, 1, 2, 3]],
        [1],
        [[0.0, 0.0, 1.0, 1.0], [0.1, 1.0, 0.9, 1.4]],
        250,
    )

    assert config["elem"] == [[1, 2, 3, 4]]
    assert config["elemprop"] == [1]
    assert config["nphoton"] == 250
    assert config["tstart"] == 0.0
    assert config["tend"] == config["tstep"] == 5e-9
    assert config["method"] == "elem"
    assert config["issaveexit"] == config["issavedet"] == 1
    assert config["outputtype"] == "flux"


def test_validate_mmc_flux_rejects_wrong_shape_and_non_finite_values() -> None:
    np.testing.assert_array_equal(validate_mmc_flux([1, 2], 2, "source 0"), [1.0, 2.0])
    with pytest.raises(ValueError, match="one value per mesh node"):
        validate_mmc_flux([1.0], 2, "source 0")
    with pytest.raises(ValueError, match="non-finite"):
        validate_mmc_flux([1.0, np.nan], 2, "source 0")


def test_mmc_to_json_returns_embedded_document() -> None:
    config = {
        "node": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "elem": [[1, 2, 3, 4]],
        "detpos": [[1.0, 2.0, 3.0, 0.5]],
        "prop": [[0.0, 0.0, 1.0, 1.0], [0.01, 1.0, 0.9, 1.4]],
        "method": "grid",
        "outputtype": "jacobian",
        "seed": 11,
    }

    document = json.loads(mmc_to_json(config))

    assert document["Session"]["RayTracer"] == "g"
    assert document["Session"]["OutputType"] == "j"
    assert document["Session"]["RNGSeed"] == 11
    assert document["Mesh"]["MeshElem"] == [[1, 2, 3, 4]]
    assert document["Optode"]["Detector"][0]["R"] == 0.5


def test_save_mmc_mesh_writes_expected_files(tmp_path) -> None:
    save_mmc_mesh(
        "head",
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        [[1, 2, 3, 4, 2]],
        tmp_path,
    )

    assert (tmp_path / "node_head.dat").read_text(encoding="utf-8").splitlines() == ["0 0 0", "1 2 3"]
    assert (tmp_path / "elem_head.dat").read_text(encoding="utf-8").strip() == "1 2 3 4 2"
