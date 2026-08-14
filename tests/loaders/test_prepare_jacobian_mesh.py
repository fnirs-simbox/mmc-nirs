import numpy as np
import pytest

from mmc_nirs.loaders.prepare_jacobian_mesh import prepare_jacobian_mesh


def test_prepare_jacobian_mesh_returns_mesh_without_saving() -> None:
    prepared = prepare_jacobian_mesh(
        np.zeros((4, 3)),
        [[0, 1, 2, 3]],
        [1, 2, 3, 4],
        orientation="RAS",
        units="mm",
    )

    assert set(prepared) == {"nodes", "elements", "node_tissue_values"}
    np.testing.assert_array_equal(prepared["elements"], [[0, 1, 2, 3]])
    np.testing.assert_array_equal(prepared["node_tissue_values"], [1, 2, 3, 4])


def test_prepare_jacobian_mesh_normalizes_and_saves_mesh(tmp_path) -> None:
    output = tmp_path / "mesh.npz"
    nodes = np.array(
        [
            [0.001, 0.002, 0.003],
            [0.004, 0.005, 0.006],
            [0.007, 0.008, 0.009],
            [0.010, 0.011, 0.012],
        ]
    )

    returned = prepare_jacobian_mesh(
        nodes,
        [[1, 2, 3, 4]],
        [50, 40, 30, 20],
        orientation="LIA",
        units="m",
        scalp_idx=10,
        skull_idx=20,
        CSF_idx=30,
        gray_matter_idx=40,
        white_matter_idx=50,
        save_mesh=True,
        filename=output,
    )

    assert returned is None
    with np.load(output) as mesh:
        np.testing.assert_allclose(mesh["nodes"], nodes * 1000 @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]).T)
        np.testing.assert_array_equal(mesh["elements"], [[0, 1, 2, 3]])
        np.testing.assert_array_equal(mesh["node_tissue_values"], [1, 2, 3, 4])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("nodes", [[0, 0]], "nodes"),
        ("elements", [[0, 1, 2]], "elements"),
        ("node_tissue_values", [1, 2, 3], "node_tissue_values"),
        ("orientation", "XYZ", "orientation"),
        ("units", "km", "units"),
    ],
)
def test_prepare_jacobian_mesh_rejects_invalid_input(field, value, message) -> None:
    arguments = {
        "nodes": np.zeros((4, 3)),
        "elements": [[0, 1, 2, 3]],
        "node_tissue_values": [1, 2, 3, 4],
        "orientation": "RAS",
        "units": "mm",
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        prepare_jacobian_mesh(**arguments)


def test_prepare_jacobian_mesh_rejects_duplicate_or_unknown_tissue_labels() -> None:
    arguments = (
        np.zeros((4, 3)),
        [[0, 1, 2, 3]],
        [1, 2, 3, 99],
        "RAS",
        "mm",
    )
    with pytest.raises(ValueError, match="unknown tissue labels"):
        prepare_jacobian_mesh(*arguments)

    with pytest.raises(ValueError, match="must be unique"):
        prepare_jacobian_mesh(*arguments, scalp_idx=1)


def test_prepare_jacobian_mesh_requires_npz_filename_only_when_saving(tmp_path) -> None:
    arguments = (np.zeros((4, 3)), [[0, 1, 2, 3]], [1, 2, 3, 4], "RAS", "mm")

    prepare_jacobian_mesh(*arguments)
    with pytest.raises(ValueError, match="filename is required"):
        prepare_jacobian_mesh(*arguments, save_mesh=True)
    with pytest.raises(ValueError, match=".npz"):
        prepare_jacobian_mesh(*arguments, save_mesh=True, filename=tmp_path / "mesh.mat")
