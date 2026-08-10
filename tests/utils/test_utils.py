import json

import numpy as np
import pytest

from mmc_nirs.utils import find_closest_node, mmc_to_json, save_mmc_mesh


def test_find_closest_node_returns_index_and_coordinates() -> None:
    nodes = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [5.0, 5.0, 5.0]])

    index, node = find_closest_node(nodes, [1.8, 1.1, 0.0])

    assert index == 1
    np.testing.assert_array_equal(node, nodes[1])


def test_find_closest_node_validates_target_shape() -> None:
    with pytest.raises(ValueError, match="one coordinate"):
        find_closest_node(np.zeros((2, 3)), [0.0, 0.0])


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
