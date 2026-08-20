import matplotlib
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Line3DCollection

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mmc_nirs.utils.plot_jacobians import _select_channel_values, plot_tissue_sensitivity


@pytest.fixture
def prepared_mesh() -> dict[str, np.ndarray]:
    return {
        "nodes": np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [3, 0, 0],
                [4, 0, 0],
                [3, 1, 0],
                [3, 0, 1],
            ],
            dtype=float,
        ),
        "elements": np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
        "element_tissue_ids": np.array([2, 2]),
        "ordered_tissue_ids": np.array([0, 1, 2]),
        "ordered_tissues": np.array(["ambient_air", "white_matter", "gray_matter"]),
    }


@pytest.fixture
def prepared_probe() -> dict[str, np.ndarray]:
    return {
        "sourcepos": np.array([[0.1, 0.1, 0.1], [3.1, 0.1, 0.1]]),
        "detpos": np.array([[0.1, 0.2, 0.1], [3.1, 0.2, 0.1]]),
        "sourcedir": np.array([[1, 0, 0], [1, 0, 0]]),
        "detnorms": np.array([[0, 1, 0], [0, 1, 0]]),
        "source_elements": np.array([0, 1]),
        "detector_elements": np.array([0, 1]),
        "channel_pairings": np.array([[0, 1], [1, 0]]),
    }


@pytest.fixture
def jacobian() -> np.ndarray:
    return np.arange(32, dtype=float).reshape(4, 8) + 1


def test_select_channel_values_maps_pairing_index_to_source_major_row(jacobian) -> None:
    values, pairings = _select_channel_values(
        jacobian,
        np.array([[0, 1], [1, 0]]),
        1,
        source_count=2,
        detector_count=2,
    )

    np.testing.assert_array_equal(values, jacobian[2])
    np.testing.assert_array_equal(pairings, [[1, 0]])


def test_select_all_averages_only_configured_channel_rows(jacobian) -> None:
    pairings = np.array([[0, 1], [1, 0], [0, 1]])

    values, selected_pairings = _select_channel_values(
        jacobian,
        pairings,
        "all",
        source_count=2,
        detector_count=2,
    )

    np.testing.assert_array_equal(values, jacobian[[1, 2]].mean(axis=0))
    np.testing.assert_array_equal(selected_pairings, [[0, 1], [1, 0]])


def test_plot_all_saves_two_panel_figure_with_every_active_channel(
    tmp_path,
    prepared_mesh,
    prepared_probe,
    jacobian,
) -> None:
    figure = plot_tissue_sensitivity(
        prepared_mesh,
        prepared_probe,
        jacobian,
        "all",
        tmp_path,
        save_filename="jacobian_690_tissue_sensitivity.png",
    )

    assert (tmp_path / "jacobian_690_tissue_sensitivity.png").is_file()
    assert [axis.get_title() for axis in figure.axes] == [r"$\gamma = 0.25$", r"$\gamma = 1.0$"]
    for axis in figure.axes:
        channel_collection = next(
            collection for collection in axis.collections if isinstance(collection, Line3DCollection)
        )
        assert len(channel_collection._segments3d) == 2
    plt.close(figure)


@pytest.mark.parametrize("channel_selection", [-1, 2, True, "configured"])
def test_plot_rejects_invalid_channel_selection(
    tmp_path,
    prepared_mesh,
    prepared_probe,
    jacobian,
    channel_selection,
) -> None:
    with pytest.raises(ValueError, match="channel_selection"):
        plot_tissue_sensitivity(
            prepared_mesh,
            prepared_probe,
            jacobian,
            channel_selection,
            tmp_path,
        )
