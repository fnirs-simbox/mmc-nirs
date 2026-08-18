import importlib

import h5py
import numpy as np
import pytest

from mmc_nirs.light_transport.prepare_jacobian_mesh import prepare_jacobian_mesh
from mmc_nirs.loaders import load_light_transport_results

probe_module = importlib.import_module("mmc_nirs.light_transport.prepare_jacobian_probe")
probe_utils_module = importlib.import_module("mmc_nirs.utils.probe_utils")
prepare_jacobian_probe = probe_module.prepare_jacobian_probe


@pytest.fixture
def experiment_config(tmp_path):
    return {
        "experiment_dir": tmp_path / "prepared",
        "filepaths": {
            "meshfile": "mesh.npz",
            "nodes_var": "nodes",
            "probefile": "probe.npz",
        },
    }


@pytest.fixture
def prepared_mesh():
    return {
        "nodes": np.array([[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, 20]], dtype=float),
        "elements": np.array([[0, 1, 2, 3]]),
        "element_tissue_values": np.array([1]),
    }


@pytest.fixture
def registration_result():
    return (
        np.array([[0, 0, 0], [10, 0, 0]], dtype=float),
        np.array([[1, 0, 0], [30, 0, 0]], dtype=float),
        np.array([[1, 0, 0], [1, 0, 0]], dtype=float),
        np.array([[-1, 0, 0], [-1, 0, 0]], dtype=float),
        np.array([0, 0]),
        np.array([0, 0]),
    )


@pytest.fixture
def fake_registration(monkeypatch, registration_result):
    calls = []

    def fake_register_probe(*args, **kwargs):
        calls.append((args, kwargs))
        return registration_result

    monkeypatch.setattr(probe_module, "register_probe", fake_register_probe)
    return calls


def test_prepare_jacobian_probe_registers_with_prepared_mesh(
    experiment_config,
    prepared_mesh,
    registration_result,
    fake_registration,
) -> None:
    sources = [[0, 0, 0], [1, 0, 0]]
    detectors = [[0.1, 0, 0], [3, 0, 0]]
    pairings = [[1, 1], [2, 2]]

    probe = prepare_jacobian_probe(
        sources,
        detectors,
        prepared_mesh,
        "cm",
        "LIA",
        pairings,
        "distance",
        15.0,
        experiment_config,
    )

    assert set(probe) == {
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
    for key, value in zip(
        ("sourcepos", "detpos", "sourcedir", "detnorms", "source_elements", "detector_elements"),
        registration_result,
        strict=True,
    ):
        np.testing.assert_array_equal(probe[key], value)
    np.testing.assert_array_equal(probe["short_separation_indices"], [0])
    np.testing.assert_array_equal(probe["long_separation_indices"], [1])
    np.testing.assert_array_equal(probe["channel_pairings"], [[0, 0], [1, 1]])

    args, kwargs = fake_registration[0]
    assert args[0] is sources
    assert args[1] is detectors
    assert args[2] is prepared_mesh["nodes"]
    assert args[3] is prepared_mesh["elements"]
    assert kwargs["probe_units"] == "cm"
    assert kwargs["probe_orientation"] == "LIA"


def test_prepare_jacobian_probe_reuses_existing_archive_before_registration(
    experiment_config,
    monkeypatch,
) -> None:
    output_dir = experiment_config["experiment_dir"]
    output_dir.mkdir()
    cached = {
        "sourcepos": np.ones((1, 3)),
        "detpos": np.ones((1, 3)),
        "sourcedir": np.ones((1, 3)),
        "detnorms": np.ones((1, 3)),
        "source_elements": np.array([0]),
        "detector_elements": np.array([0]),
        "channel_pairings": np.array([[0, 0]]),
        "short_separation_indices": np.array([0]),
        "long_separation_indices": np.array([], dtype=int),
    }
    np.savez(output_dir / "probe.npz", **cached)
    monkeypatch.setattr(probe_module, "register_probe", lambda *args, **kwargs: pytest.fail("registration ran"))

    probe = prepare_jacobian_probe([], [], {}, "invalid", "invalid", [], "invalid", object(), experiment_config)

    for key, value in cached.items():
        np.testing.assert_array_equal(probe[key], value)


def test_prepare_jacobian_probe_overwrites_existing_archive(
    experiment_config,
    prepared_mesh,
    registration_result,
    fake_registration,
) -> None:
    output_dir = experiment_config["experiment_dir"]
    output_dir.mkdir()
    cached = {key: np.zeros(1) for key in probe_module._PROBE_ARCHIVE_KEYS}
    np.savez(output_dir / "probe.npz", **cached)

    probe = prepare_jacobian_probe(
        [[0, 0, 0], [1, 0, 0]],
        [[0.1, 0, 0], [3, 0, 0]],
        prepared_mesh,
        "cm",
        "RAS",
        [[1, 1], [2, 2]],
        "index",
        [0],
        experiment_config,
        save_probe=True,
        overwrite=True,
    )

    assert len(fake_registration) == 1
    np.testing.assert_array_equal(probe["sourcepos"], registration_result[0])
    np.testing.assert_array_equal(probe["long_separation_indices"], [1])
    with np.load(output_dir / "probe.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["sourcepos"], registration_result[0])
        np.testing.assert_array_equal(archive["long_separation_indices"], [1])


def test_prepare_jacobian_probe_rejects_incompatible_cache(experiment_config) -> None:
    output_dir = experiment_config["experiment_dir"]
    output_dir.mkdir()
    np.savez(output_dir / "probe.npz", sourcepos=np.zeros((1, 3)))

    with pytest.raises(ValueError, match="missing required field"):
        prepare_jacobian_probe([], [], {}, "mm", "RAS", [], "index", [], experiment_config)


@pytest.mark.parametrize(
    ("pairings", "flag", "argument", "message"),
    [
        ([[0, 0]], "invalid", [0], "short_separation_flag"),
        ([[0, 0]], "distance", 1, "finite float"),
        ([[0, 0]], "index", (0,), "list of integers"),
        ([[0, 0]], "index", [1], "out of range"),
        ([], "index", [0], "channel_pairings"),
        ([[0, 0, 0]], "index", [0], "channel_pairings"),
    ],
)
def test_prepare_jacobian_probe_rejects_invalid_channel_configuration(
    experiment_config,
    prepared_mesh,
    pairings,
    flag,
    argument,
    message,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        prepare_jacobian_probe(
            [[0, 0, 0]],
            [[1, 0, 0]],
            prepared_mesh,
            "mm",
            "RAS",
            pairings,
            flag,
            argument,
            experiment_config,
        )


def test_saved_prepared_inputs_are_loadable_downstream(
    experiment_config,
    registration_result,
    fake_registration,
) -> None:
    mesh = prepare_jacobian_mesh(
        np.array([[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, 20]], dtype=float),
        [[0, 1, 2, 3]],
        [1],
        "RAS",
        "mm",
        experiment_config,
        save_mesh=True,
    )
    prepare_jacobian_probe(
        [[0, 0, 0], [1, 0, 0]],
        [[0.1, 0, 0], [3, 0, 0]],
        mesh,
        "cm",
        "RAS",
        [[1, 1], [2, 2]],
        "index",
        [0],
        experiment_config,
        save_probe=True,
    )

    loaded = load_light_transport_results(experiment_config, use_jacobian=False)

    np.testing.assert_array_equal(loaded["nodes"], mesh["nodes"])
    np.testing.assert_array_equal(loaded["source_positions"], registration_result[0])
    np.testing.assert_array_equal(loaded["detector_positions"], registration_result[1])
    np.testing.assert_array_equal(loaded["detector_norms"], registration_result[3])


def test_load_channel_pairs_from_snirf_sorts_measurement_lists_numerically(tmp_path) -> None:
    snirf_path = tmp_path / "probe.snirf"
    with h5py.File(snirf_path, "w") as snirf:
        data_group = snirf.create_group("nirs").create_group("data1")
        for measurement_number, source_index, detector_index in ((10, 3, 4), (2, 1, 2)):
            measurement = data_group.create_group(f"measurementList{measurement_number}")
            measurement.create_dataset("sourceIndex", data=[source_index])
            measurement.create_dataset("detectorIndex", data=[detector_index])

    pairings = probe_module.load_channel_pairs_from_snirf(snirf_path)

    np.testing.assert_array_equal(pairings, [[1, 2], [3, 4]])


def test_signed_surface_distances_are_negative_inside_and_positive_outside() -> None:
    nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    elements = np.array([[0, 1, 2, 3]])

    distances = probe_utils_module._signed_surface_distances(
        np.array([[1.0, 1.0, 1.0], [10.0, 10.0, 10.0]]),
        nodes,
        elements,
    )

    assert distances[0] < 0
    assert distances[1] > 0
