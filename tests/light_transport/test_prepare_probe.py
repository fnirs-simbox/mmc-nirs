import importlib

import numpy as np
import pytest

from mmcnirs.light_transport.prepare_mesh import prepare_mesh

probe_module = importlib.import_module("mmcnirs.light_transport.prepare_probe")
probe_utils_module = importlib.import_module("mmcnirs.utils.probe_utils")
prepare_probe = probe_module.prepare_probe


@pytest.fixture
def experiment_config(tmp_path):
    return {
        "experiment_dir": tmp_path / "prepared",
        "mesh_settings": {
            "ordered_tissues": {"0": "ambient_air", "1": "tissue"},
            "mesh_orientation": "RAS",
            "mesh_units": "mm",
        },
        "probe_settings": {
            "probe_units": "cm",
            "probe_orientation": "LIA",
            "short_separation_flag": "distance",
            "short_separation_arg": 15.0,
            "embedding_step": 0.5,
            "max_embedding_steps": 1_000,
        },
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
        "element_tissue_ids": np.array([1]),
        "ordered_tissue_ids": np.array([0, 1]),
        "ordered_tissues": np.array(["ambient_air", "tissue"]),
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


def test_prepare_probe_registers_with_prepared_mesh(
    experiment_config,
    prepared_mesh,
    registration_result,
    fake_registration,
) -> None:
    sources = [[0, 0, 0], [1, 0, 0]]
    detectors = [[0.1, 0, 0], [3, 0, 0]]
    pairings = [[1, 1], [2, 2]]

    probe = prepare_probe(
        sources,
        detectors,
        prepared_mesh,
        pairings,
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
    np.testing.assert_array_equal(args[2], prepared_mesh["nodes"])
    np.testing.assert_array_equal(args[3], prepared_mesh["elements"])
    assert args[2] is not prepared_mesh["nodes"]
    assert args[3] is not prepared_mesh["elements"]
    assert kwargs["probe_units"] == "cm"
    assert kwargs["probe_orientation"] == "LIA"
    assert kwargs["embedding_step"] == 0.5
    assert kwargs["max_embedding_steps"] == 1_000


def test_prepare_probe_reuses_existing_archive_before_registration(
    experiment_config,
    prepared_mesh,
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

    probe = prepare_probe([], [], prepared_mesh, [], experiment_config)

    for key, value in cached.items():
        np.testing.assert_array_equal(probe[key], value)


def test_prepare_probe_saves_diagnostic_beside_cached_probe(
    experiment_config,
    prepared_mesh,
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
    saved_paths = []

    class FakeFigure:
        def savefig(self, path):
            saved_paths.append(path)

    monkeypatch.setattr(probe_module, "_plot_probe_registration", lambda *args: FakeFigure())

    prepare_probe([], [], prepared_mesh, [], experiment_config, plot=True)

    assert saved_paths == [output_dir / "register_probe_diagnostic.png"]


def test_prepare_probe_overwrites_existing_archive(
    experiment_config,
    prepared_mesh,
    registration_result,
    fake_registration,
) -> None:
    output_dir = experiment_config["experiment_dir"]
    output_dir.mkdir()
    cached = {key: np.zeros(1) for key in probe_module._PROBE_ARCHIVE_KEYS}
    np.savez(output_dir / "probe.npz", **cached)

    experiment_config["probe_settings"].update(
        probe_orientation="RAS",
        short_separation_flag="index",
        short_separation_arg=[0],
    )
    probe = prepare_probe(
        [[0, 0, 0], [1, 0, 0]],
        [[0.1, 0, 0], [3, 0, 0]],
        prepared_mesh,
        [[1, 1], [2, 2]],
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


def test_prepare_probe_rejects_incompatible_cache(experiment_config, prepared_mesh) -> None:
    output_dir = experiment_config["experiment_dir"]
    output_dir.mkdir()
    np.savez(output_dir / "probe.npz", sourcepos=np.zeros((1, 3)))

    with pytest.raises(ValueError, match="missing required field"):
        prepare_probe([], [], prepared_mesh, [], experiment_config)


def test_prepare_probe_validates_mesh_before_cache_reuse(experiment_config) -> None:
    output_dir = experiment_config["experiment_dir"]
    output_dir.mkdir()
    cached = {key: np.zeros(1) for key in probe_module._PROBE_ARCHIVE_KEYS}
    np.savez(output_dir / "probe.npz", **cached)

    with pytest.raises(ValueError, match="prepared_mesh is missing required field"):
        prepare_probe([], [], {}, [], experiment_config)


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
def test_prepare_probe_rejects_invalid_channel_configuration(
    experiment_config,
    prepared_mesh,
    pairings,
    flag,
    argument,
    message,
) -> None:
    experiment_config["probe_settings"]["short_separation_flag"] = flag
    experiment_config["probe_settings"]["short_separation_arg"] = argument
    with pytest.raises((TypeError, ValueError), match=message):
        prepare_probe(
            [[0, 0, 0]],
            [[1, 0, 0]],
            prepared_mesh,
            pairings,
            experiment_config,
        )


def test_saved_prepared_inputs_are_loadable_downstream(
    experiment_config,
    registration_result,
    fake_registration,
) -> None:
    experiment_config["filepaths"]["meshfile"] = "mmcnirs_outputs/mesh.npz"
    experiment_config["filepaths"]["probefile"] = "mmcnirs_outputs/probe.npz"
    mesh = prepare_mesh(
        np.array([[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, 20]], dtype=float),
        [[0, 1, 2, 3]],
        [1],
        experiment_config,
        save_mesh=True,
    )
    experiment_config["probe_settings"].update(
        probe_orientation="RAS",
        short_separation_flag="index",
        short_separation_arg=[0],
    )
    prepare_probe(
        [[0, 0, 0], [1, 0, 0]],
        [[0.1, 0, 0], [3, 0, 0]],
        mesh,
        [[1, 1], [2, 2]],
        experiment_config,
        save_probe=True,
    )

    mesh_path = experiment_config["experiment_dir"] / "mmcnirs_outputs" / "mesh.npz"
    probe_path = experiment_config["experiment_dir"] / "mmcnirs_outputs" / "probe.npz"
    assert mesh_path.is_file()
    assert probe_path.is_file()

    with np.load(mesh_path, allow_pickle=False) as mesh_archive:
        np.testing.assert_array_equal(mesh_archive["nodes"], mesh["nodes"])
    with np.load(probe_path, allow_pickle=False) as probe_archive:
        np.testing.assert_array_equal(probe_archive["sourcepos"], registration_result[0])
        np.testing.assert_array_equal(probe_archive["detpos"], registration_result[1])
        np.testing.assert_array_equal(probe_archive["detnorms"], registration_result[3])


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


def test_prepare_probe_requires_complete_settings_before_cache_reuse(experiment_config) -> None:
    output_dir = experiment_config["experiment_dir"]
    output_dir.mkdir()
    np.savez(output_dir / "probe.npz", **{key: np.zeros(1) for key in probe_module._PROBE_ARCHIVE_KEYS})
    del experiment_config["probe_settings"]["probe_units"]
    del experiment_config["probe_settings"]["embedding_step"]

    with pytest.raises(ValueError) as error:
        prepare_probe([], [], {}, [], experiment_config)

    message = str(error.value)
    assert (
        "required keys: embedding_step, max_embedding_steps, probe_orientation, probe_units, "
        "short_separation_arg, short_separation_flag" in message
    )
    assert "missing keys: embedding_step, probe_units" in message


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("probe_units", "km", "probe_units"),
        ("probe_orientation", "XYZ", "probe orientation"),
        ("embedding_step", 0, "embedding_step"),
        ("embedding_step", np.inf, "embedding_step"),
        ("max_embedding_steps", -1, "max_embedding_steps"),
        ("max_embedding_steps", 1.5, "max_embedding_steps"),
    ],
)
def test_prepare_probe_validates_settings_before_cache_reuse(
    experiment_config,
    field,
    value,
    message,
) -> None:
    output_dir = experiment_config["experiment_dir"]
    output_dir.mkdir()
    np.savez(output_dir / "probe.npz", **{key: np.zeros(1) for key in probe_module._PROBE_ARCHIVE_KEYS})
    experiment_config["probe_settings"][field] = value

    with pytest.raises(ValueError, match=message):
        prepare_probe([], [], {}, [], experiment_config)
