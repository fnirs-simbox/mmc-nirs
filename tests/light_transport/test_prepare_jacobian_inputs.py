import numpy as np
import pytest

from mmcnirs.light_transport.prepare_jacobian_inputs import prepare_jacobian_inputs


def test_prepare_jacobian_inputs_normalizes_all_generator_inputs(jacobian_inputs) -> None:
    prepared = prepare_jacobian_inputs(*jacobian_inputs, {"nphoton": 2e9}, 690)

    np.testing.assert_array_equal(prepared.elements, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(prepared.element_tissue_ids, [1])
    np.testing.assert_array_equal(prepared.channel_indices, [0, 2, 3])
    np.testing.assert_array_equal(prepared.closest_detector_nodes, [0, 1])
    np.testing.assert_array_equal(
        prepared.selected_properties,
        [[0.0, 0.0, 1.0, 1.0], [0.1, 1.0, 0.9, 1.4]],
    )
    assert prepared.photon_count == 2_000_000_000


@pytest.mark.parametrize(
    ("mesh_update", "probe_update", "message"),
    [
        ({"element_tissue_ids": np.array([2])}, {}, "not represented"),
        ({"elements": np.array([[1, 2, 3, 4]])}, {}, "out-of-range vertex"),
        ({}, {"source_elements": np.array([1, 0])}, "out-of-range element"),
        ({}, {"sourcedir": np.ones((1, 3))}, "must match sourcepos"),
        (
            {},
            {"detnorms": np.array([[0.0, 0.0, 2.0], [-1.0, 0.0, 0.0]])},
            "prepare_probe may not have been run",
        ),
        ({}, {"channel_pairings": np.array([[0, 2]])}, "out-of-range detector"),
        ({}, {"channel_pairings": np.array([[1, 1], [2, 2]])}, "out-of-range source"),
    ],
)
def test_prepare_jacobian_inputs_rejects_incompatible_prepared_data(
    jacobian_inputs,
    mesh_update,
    probe_update,
    message,
) -> None:
    mesh, probe, properties = jacobian_inputs

    with pytest.raises(ValueError, match=message):
        prepare_jacobian_inputs(
            mesh | mesh_update,
            probe | probe_update,
            properties,
            {"nphoton": 100},
            690,
        )


def test_prepare_jacobian_inputs_rejects_missing_wavelength(jacobian_inputs) -> None:
    with pytest.raises(ValueError, match="No optical properties"):
        prepare_jacobian_inputs(*jacobian_inputs, {"nphoton": 100}, 850)
