"""Low-level MMC serialization and output helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike


def _as_list(value: ArrayLike, dtype: DTypeLike | None = None) -> list[Any]:
    return np.asarray(value, dtype=dtype).tolist()


def _copy_config_value(
    source: Mapping[str, Any],
    source_key: str,
    target: dict[str, Any],
    target_key: str,
    default: Any = None,
) -> None:
    if source_key in source:
        target[target_key] = source[source_key]
    elif default is not None:
        target[target_key] = default


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


def read_cli_output(file_stub: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read fluence and detected-photon output from an MMC CLI run.

    Parameters
    ----------
    file_stub : str or pathlib.Path
        Common path without the ``.dat`` or ``.mch`` suffix.

    Returns
    -------
    flux : numpy.ndarray
        Flux values from the second column of the ``.dat`` file.
    detected_photons : dict[str, numpy.ndarray]
        Detector IDs, scattering counts, partial paths, exit positions, and exit
        directions parsed from the ``.mch`` file.

    Raises
    ------
    ValueError
        If an output file does not contain the expected columns.
    """
    import pmmc

    stub = Path(file_stub)
    flux_data = np.loadtxt(stub.with_suffix(".dat"), ndmin=2)
    if flux_data.shape[1] < 2:
        raise ValueError("MMC flux output must contain at least two columns")

    photon_data, metadata = pmmc.loadmch(str(stub.with_suffix(".mch")))
    medium_count = int(metadata["medianum"])
    expected_columns = 1 + 2 * medium_count + 6
    if photon_data.ndim != 2 or photon_data.shape[1] < expected_columns:
        raise ValueError(f"MMC photon output must contain at least {expected_columns} columns")

    scattering_end = 1 + medium_count
    path_end = scattering_end + medium_count
    position_end = path_end + 3
    direction_end = position_end + 3
    detected_photons = {
        "detector_id": photon_data[:, 0],
        "scattering_counts": photon_data[:, 1:scattering_end],
        "partial_paths": photon_data[:, scattering_end:path_end],
        "exit_positions": photon_data[:, path_end:position_end],
        "exit_directions": photon_data[:, position_end:direction_end],
    }
    return flux_data[:, 1], detected_photons


def find_closest_node(nodes: ArrayLike, target_point: ArrayLike) -> tuple[int, np.ndarray]:
    """Find the mesh node closest to a target point in Euclidean distance.

    Parameters
    ----------
    nodes : array-like
        Two-dimensional node coordinate array with shape ``(n_nodes, n_dimensions)``.
    target_point : array-like
        Coordinates of one target point with shape ``(n_dimensions,)``.

    Returns
    -------
    index : int
        Index of the closest node.
    node : numpy.ndarray
        Coordinates of the closest node.

    Raises
    ------
    ValueError
        If the input shapes are incompatible or no nodes are provided.
    """
    node_array = np.asarray(nodes)
    target = np.asarray(target_point)
    if node_array.ndim != 2 or node_array.shape[0] == 0:
        raise ValueError("nodes must be a non-empty two-dimensional array")
    if target.shape != (node_array.shape[1],):
        raise ValueError("target_point must have one coordinate per node dimension")

    squared_distances = np.sum((node_array - target) ** 2, axis=1)
    closest_index = int(np.argmin(squared_distances))
    return closest_index, node_array[closest_index]
