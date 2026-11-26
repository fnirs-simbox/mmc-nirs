import numpy as np
import pytest


@pytest.fixture
def jacobian_inputs():
    mesh = {
        "nodes": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        "elements": np.array([[0, 1, 2, 3]]),
        "node_tissue_ids": np.ones(4, dtype=int),
        "element_tissue_ids": np.array([1]),
        "ordered_tissue_ids": np.array([0, 1]),
        "ordered_tissues": np.array(["ambient_air", "tissue"]),
    }
    probe = {
        "sourcepos": np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]]),
        "detpos": np.array([[0.0, 0.0, 0.0], [0.625, 0.25, 0.0]]),
        "sourcedir": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "detnorms": np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]),
        "source_elements": np.array([0, 0]),
        "detector_elements": np.array([0, 0]),
        "channel_pairings": np.array([[0, 0], [0, 0], [1, 0], [1, 1]]),
    }
    optical_properties = {"690": {"tissue": [0.1, 1.0, 0.9, 1.4], "ambient_air": [0.0, 0.0, 1.0, 1.0]}}
    return mesh, probe, optical_properties
