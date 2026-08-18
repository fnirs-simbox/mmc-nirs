"""Low-level MMC serialization and Jacobian-generation helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .prepared_input_io import load_npz_archive, require_fields, save_npz_archive

JACOBIAN_TSTEP_SECONDS = 5e-9
MMC_SETTING_KEYS = {"nphoton"}

JACOBIAN_RESULT_KEYS = {
    "Green_d",
    "Green_s",
    "Green_sd",
    "J",
    "channelidx",
    "mea0",
    "sourcepos",
    "detpos",
    "detnorms",
    "sourcedir",
}


def build_jacobian_mmc_config(
    nodes: ArrayLike,
    elements: ArrayLike,
    element_tissue_values: ArrayLike,
    optical_properties: ArrayLike,
    photon_count: int,
) -> dict[str, Any]:
    """Build the fixed legacy MMC configuration from canonical inputs."""
    return {
        "nphoton": photon_count,
        "node": np.asarray(nodes).tolist(),
        "elem": (np.asarray(elements) + 1).tolist(),
        "elemprop": np.asarray(element_tissue_values).tolist(),
        "tstart": 0.0,
        "tend": JACOBIAN_TSTEP_SECONDS,
        "tstep": JACOBIAN_TSTEP_SECONDS,
        "prop": np.asarray(optical_properties).tolist(),
        "method": "elem",
        "issaveexit": 1,
        "issavedet": 1,
        "outputtype": "flux",
    }


def validate_mmc_flux(flux: ArrayLike, node_count: int, description: str) -> np.ndarray:
    """Return an MMC flux vector after validating its node dimension."""
    flux_array = np.asarray(flux, dtype=float)
    if flux_array.shape != (node_count,):
        raise ValueError(f"{description} flux must contain one value per mesh node")
    if not np.all(np.isfinite(flux_array)):
        raise ValueError(f"{description} flux contains non-finite values")
    return flux_array


def resolve_jacobian_save_path(save_path: str | Path | None) -> Path:
    """Validate and resolve the destination for a Jacobian archive."""
    if save_path is None:
        raise ValueError("save_path must be provided when save=True")
    path = Path(save_path).expanduser()
    if path.suffix.lower() != ".npz":
        raise ValueError("save_path must name a .npz file")
    if path.exists() and not path.is_file():
        raise ValueError(f"save_path is not a file: {path}")
    return path


def load_jacobian_result(path: Path) -> dict[str, np.ndarray]:
    """Load a cached Jacobian archive and require all legacy result fields."""
    return load_npz_archive(path, JACOBIAN_RESULT_KEYS)


def save_jacobian_result(path: Path, result: Mapping[str, Any]) -> None:
    """Save a generated Jacobian result archive."""
    save_npz_archive(path, result)


def order_optical_properties(
    optical_properties: Mapping[str, Mapping[str, ArrayLike]],
    ordered_tissues: Sequence[str],
) -> dict[str, list[list[float]]]:
    """Arrange wavelength-specific optical properties in MMC medium order.

    Parameters
    ----------
    optical_properties : mapping
        Mapping loaded from ``optical_properties.json``. Each wavelength maps
        tissue names to ``[mua, mus, g, n]`` values.
    ordered_tissues : sequence of str
        Tissue names in the medium order expected by the prepared mesh. The
        background medium, normally ``"ambient_air"``, must be included.

    Returns
    -------
    dict[str, list[list[float]]]
        Optical-property rows ordered identically for every wavelength.

    Raises
    ------
    TypeError
        If either input does not have the expected mapping/sequence structure.
    ValueError
        If a tissue is missing, duplicated, or does not contain four finite
        optical-property values.
    """
    if not isinstance(optical_properties, Mapping) or not optical_properties:
        raise TypeError("optical_properties must be a non-empty mapping")
    if isinstance(ordered_tissues, (str, bytes)) or not isinstance(ordered_tissues, Sequence):
        raise TypeError("ordered_tissues must be a sequence of tissue names")

    tissue_order = list(ordered_tissues)
    if not tissue_order or not all(isinstance(tissue, str) and tissue for tissue in tissue_order):
        raise ValueError("ordered_tissues must contain non-empty tissue names")
    if len(set(tissue_order)) != len(tissue_order):
        raise ValueError("ordered_tissues must not contain duplicate tissue names")

    ordered_properties: dict[str, list[list[float]]] = {}
    for wavelength, tissue_properties in optical_properties.items():
        if not isinstance(wavelength, str) or not wavelength:
            raise ValueError("optical property wavelength keys must be non-empty strings")
        if not isinstance(tissue_properties, Mapping):
            raise TypeError(f"optical properties for wavelength {wavelength!r} must be a mapping")

        wavelength_properties: list[list[float]] = []
        for tissue in tissue_order:
            try:
                values = np.asarray(tissue_properties[tissue], dtype=float)
            except KeyError as error:
                raise ValueError(
                    f"optical properties for wavelength {wavelength!r} are missing tissue {tissue!r}"
                ) from error
            if values.shape != (4,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"optical properties for wavelength {wavelength!r}, tissue {tissue!r} "
                    "must contain four finite values"
                )
            wavelength_properties.append(values.tolist())

        ordered_properties[wavelength] = wavelength_properties

    return ordered_properties


def select_optical_properties(
    optical_properties: Mapping[str, Mapping[str, ArrayLike]],
    ordered_tissues: Sequence[str],
    wavelength: str | int,
) -> np.ndarray:
    """Return ordered MMC media for one validated wavelength."""
    if isinstance(wavelength, bool) or not isinstance(wavelength, (str, int, np.integer)):
        raise TypeError("wavelength must be a string or integer")
    wavelength_key = str(wavelength)
    if not wavelength_key:
        raise ValueError("wavelength must not be empty")

    properties_by_wavelength = order_optical_properties(optical_properties, ordered_tissues)
    try:
        return np.asarray(properties_by_wavelength[wavelength_key], dtype=float)
    except KeyError as error:
        available = ", ".join(properties_by_wavelength)
        raise ValueError(f"No optical properties for wavelength {wavelength_key!r}; available: {available}") from error


def validate_mmc_settings(mmc_settings: Mapping[str, Any]) -> int:
    """Validate Jacobian MMC settings and return the integer photon count."""
    if not isinstance(mmc_settings, Mapping):
        raise TypeError("mmc_settings must be a mapping")
    require_fields(mmc_settings, MMC_SETTING_KEYS, "mmc_settings")
    unexpected_settings = set(mmc_settings).difference(MMC_SETTING_KEYS)
    if unexpected_settings:
        unexpected = ", ".join(sorted(unexpected_settings))
        raise ValueError(f"mmc_settings contains unsupported field(s): {unexpected}")

    value = mmc_settings["nphoton"]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("mmc_settings['nphoton'] must be a positive integer")
    if not np.isfinite(value) or value <= 0 or not float(value).is_integer():
        raise ValueError("mmc_settings['nphoton'] must be a positive integer")
    return int(value)


def _as_list(value: ArrayLike, dtype: DTypeLike | None = None) -> list[Any]:
    """Convert an array-like value to a JSON-compatible nested list."""
    return np.asarray(value, dtype=dtype).tolist()


def _copy_config_value(
    source: Mapping[str, Any],
    source_key: str,
    target: dict[str, Any],
    target_key: str,
) -> None:
    """Copy an optional configuration value under its serialized key."""
    if source_key in source:
        target[target_key] = source[source_key]


def save_mmc_mesh(
    mesh_tag: str,
    nodes: ArrayLike,
    elements: ArrayLike,
    directory: str | Path = ".",
) -> None:
    """Write an MMC mesh to external node and element files.

    Parameters
    ----------
    mesh_tag : str
        Identifier used in ``node_<mesh_tag>.dat`` and ``elem_<mesh_tag>.dat``.
    nodes : array-like
        Two-dimensional array with at least three coordinates per node.
    elements : array-like
        Two-dimensional array with four vertex indices and an optional material
        identifier per tetrahedron.
    directory : str or pathlib.Path, default="."
        Output directory. It is created when necessary.

    Raises
    ------
    ValueError
        If the node or element arrays do not have the required shape.
    """
    node_array = np.asarray(nodes)
    element_array = np.asarray(elements)
    if node_array.ndim != 2 or node_array.shape[1] < 3:
        raise ValueError("nodes must be a two-dimensional array with at least three columns")
    if element_array.ndim != 2 or element_array.shape[1] < 4:
        raise ValueError("elements must be a two-dimensional array with at least four columns")

    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    node_path = output_directory / f"node_{mesh_tag}.dat"
    element_path = output_directory / f"elem_{mesh_tag}.dat"

    with node_path.open("w", encoding="utf-8") as node_file:
        for node in node_array:
            node_file.write(f"{node[0]:.9g} {node[1]:.9g} {node[2]:.9g}\n")

    with element_path.open("w", encoding="utf-8") as element_file:
        for element in element_array:
            values = element[:5] if element.size >= 5 else element[:4]
            element_file.write(" ".join(str(int(value)) for value in values) + "\n")


def mmc_to_json(
    config: Mapping[str, Any],
    output_path: str | Path | None = None,
    **json_options: Any,
) -> str | None:
    """Convert an MMC configuration mapping to MMC's JSON representation.

    Parameters
    ----------
    config : mapping
        MMC configuration using the field names accepted by the MATLAB and Python
        MMC interfaces.
    output_path : str, pathlib.Path, or None, default=None
        Destination JSON path. A ``.json`` suffix embeds mesh data in one file;
        a suffix-free path writes linked mesh files beside ``<output_path>.json``.
        If omitted, the embedded JSON document is returned as a string.
    **json_options
        Additional options forwarded to :func:`json.dumps` or :func:`json.dump`.

    Returns
    -------
    str or None
        JSON text when ``output_path`` is omitted; otherwise ``None``.

    Raises
    ------
    ValueError
        If a field has an invalid value or ``output_path`` has an unsupported suffix.
    """
    path = Path(output_path) if output_path is not None else None
    if path is not None and path.suffix not in {"", ".json"}:
        raise ValueError("output_path must have a .json suffix or no suffix")

    single_file = path is None or path.suffix == ".json"
    output_stem = path.with_suffix("") if path is not None else None
    session_id = output_stem.name if output_stem is not None else "session"

    source: dict[str, Any] = {}
    _copy_config_value(config, "srcpos", source, "Pos")
    _copy_config_value(config, "srcdir", source, "Dir")
    _copy_config_value(config, "srcparam1", source, "Param1")
    _copy_config_value(config, "srcparam2", source, "Param2")
    _copy_config_value(config, "srctype", source, "Type")
    _copy_config_value(config, "srcnum", source, "SrcNum")
    if config.get("srcpattern") is not None:
        source["Pattern"] = _as_list(config["srcpattern"], dtype=np.float32)

    optode: dict[str, Any] = {"Source": source}
    if config.get("detpos") is not None:
        detector_array = np.asarray(config["detpos"])
        if detector_array.ndim != 2 or detector_array.shape[1] < 4:
            raise ValueError("detpos must contain rows of x, y, z, and radius values")
        optode["Detector"] = [{"Pos": detector[:3].tolist(), "R": detector[3].item()} for detector in detector_array]

    mesh: dict[str, Any] = {}
    _copy_config_value(config, "unitinmm", mesh, "LengthUnit")
    has_mesh = config.get("node") is not None and config.get("elem") is not None
    if has_mesh:
        nodes = np.asarray(config["node"])
        elements = np.asarray(config["elem"])
        if elements.ndim != 2:
            raise ValueError("elem must be a two-dimensional array")

        if config.get("elemprop") is not None and elements.shape[1] == 4:
            element_properties = np.asarray(config["elemprop"]).reshape(-1)
            if element_properties.size != elements.shape[0]:
                raise ValueError("elemprop length must match the number of elements")
            elements = np.column_stack((elements, element_properties))

        if single_file:
            mesh["MeshNode"] = _as_list(nodes, dtype=np.float32)
            mesh["MeshElem"] = _as_list(elements, dtype=np.uint32)
            for roi_key in ("edgeroi", "noderoi", "faceroi"):
                if roi_key in config:
                    mesh["MeshROI"] = _as_list(config[roi_key], dtype=np.float32)
                    break
        else:
            mesh["MeshID"] = session_id
            save_mmc_mesh(session_id, nodes, elements, output_stem.parent)
        _copy_config_value(config, "e0", mesh, "InitElem")
    else:
        mesh["MeshID"] = session_id or "mesh"

    domain: dict[str, Any] = {}
    _copy_config_value(config, "steps", domain, "Step")
    if config.get("prop") is not None:
        properties = np.asarray(config["prop"])
        if properties.ndim != 2 or properties.shape[1] < 4:
            raise ValueError("prop must be a two-dimensional array with at least four columns")
        if single_file:
            domain["Media"] = [
                {"mua": row[0].item(), "mus": row[1].item(), "g": row[2].item(), "n": row[3].item()}
                for row in properties
            ]
        else:
            if properties.shape[0] < 2:
                raise ValueError("prop must contain a background row and at least one medium")
            property_path = output_stem.parent / f"prop_{session_id}.dat"
            with property_path.open("w", encoding="utf-8") as property_file:
                property_file.write(f"1 {properties.shape[0] - 1}\n")
                for medium_index, row in enumerate(properties[1:], start=1):
                    property_file.write(f"{medium_index} {row[0]:.6e} {row[1]:.6e} {row[2]:.6e} {row[3]:.6e}\n")

    session: dict[str, Any] = {"ID": session_id}
    session_key_map = {
        "isreflect": "DoMismatch",
        "issave2pt": "DoSaveVolume",
        "issavedet": "DoPartialPath",
        "issaveexit": "DoSaveExit",
        "issaveseed": "DoSaveSeed",
        "isnormalized": "DoNormalize",
        "ismomentum": "DoDCS",
        "isspecular": "DoSpecular",
        "outputformat": "OutputFormat",
        "debuglevel": "DebugFlag",
        "autopilot": "DoAutoThread",
        "basisorder": "BasisOrder",
        "nphoton": "Photons",
    }
    for source_key, target_key in session_key_map.items():
        _copy_config_value(config, source_key, session, target_key)

    method_codes = {"plucker": "p", "havel": "h", "badouel": "b", "elem": "s", "grid": "g"}
    if "method" in config:
        try:
            session["RayTracer"] = method_codes[config["method"]]
        except KeyError as error:
            raise ValueError("method must be one of: plucker, havel, badouel, elem, grid") from error

    output_type_codes = {"flux": "x", "fluence": "f", "energy": "e", "jacobian": "j", "wl": "l", "wp": "p"}
    if "outputtype" in config:
        try:
            session["OutputType"] = output_type_codes[config["outputtype"]]
        except KeyError as error:
            raise ValueError("outputtype must be one of: flux, fluence, energy, jacobian, wl, wp") from error

    seed = config.get("seed")
    if isinstance(seed, int) or isinstance(seed, float) and seed.is_integer():
        session["RNGSeed"] = int(seed)

    forward = {
        "T0": config.get("tstart", 0.0),
        "T1": config.get("tend", 0.0),
        "Dt": config.get("tstep", 0.0),
    }
    _copy_config_value(config, "nout", forward, "N0")

    mmc_session = {
        "Session": session,
        "Domain": domain,
        "Mesh": mesh,
        "Forward": forward,
        "Optode": optode,
    }
    serialization_options = {"ensure_ascii": False, "indent": 2} | json_options
    if path is None:
        return json.dumps(mmc_session, **serialization_options)

    json_path = path if path.suffix == ".json" else path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(mmc_session, output_file, **serialization_options)
    return None
