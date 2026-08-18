"""Validate and normalize prepared inputs for MMC Jacobian generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mmc_nirs.utils.jacobian_utils import select_optical_properties, validate_mmc_settings
from mmc_nirs.utils.mesh_utils import (
    find_closest_nodes,
    validate_prepared_mesh,
    validate_tissue_property_coverage,
)
from mmc_nirs.utils.probe_utils import flatten_channel_pairings, validate_prepared_probe


@dataclass(frozen=True)
class JacobianInputs:
    """Canonical arrays and settings consumed by Jacobian generation."""

    nodes: np.ndarray
    elements: np.ndarray
    element_tissue_values: np.ndarray
    source_positions: np.ndarray
    detector_positions: np.ndarray
    source_directions: np.ndarray
    detector_directions: np.ndarray
    source_elements: np.ndarray
    detector_elements: np.ndarray
    selected_properties: np.ndarray
    channel_indices: np.ndarray
    closest_detector_nodes: np.ndarray
    photon_count: int


def prepare_jacobian_inputs(
    prepared_mesh: Mapping[str, ArrayLike],
    prepared_probe: Mapping[str, ArrayLike],
    optical_properties: Mapping[str, Mapping[str, ArrayLike]],
    ordered_tissues: Sequence[str],
    mmc_settings: Mapping[str, Any],
    wavelength: str | int,
) -> JacobianInputs:
    """Return validated, canonical inputs for one Jacobian wavelength.

    Mesh and probe preparation and registration must already be complete. The
    prepared mesh uses zero-based tetrahedra plus one tissue label per element;
    prepared optode element indices are also zero-based.
    """
    mesh = validate_prepared_mesh(prepared_mesh)
    probe = validate_prepared_probe(prepared_probe, len(mesh["elements"]))
    selected_properties = select_optical_properties(optical_properties, ordered_tissues, wavelength)
    validate_tissue_property_coverage(mesh["element_tissue_values"], len(selected_properties))
    channel_indices = flatten_channel_pairings(
        probe["channel_pairings"],
        len(probe["sourcepos"]),
        len(probe["detpos"]),
    )
    return JacobianInputs(
        nodes=mesh["nodes"],
        elements=mesh["elements"],
        element_tissue_values=mesh["element_tissue_values"],
        source_positions=probe["sourcepos"],
        detector_positions=probe["detpos"],
        source_directions=probe["sourcedir"],
        detector_directions=probe["detnorms"],
        source_elements=probe["source_elements"],
        detector_elements=probe["detector_elements"],
        selected_properties=selected_properties,
        channel_indices=channel_indices,
        closest_detector_nodes=find_closest_nodes(mesh["nodes"], probe["detpos"]),
        photon_count=validate_mmc_settings(mmc_settings),
    )
