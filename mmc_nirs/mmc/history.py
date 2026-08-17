"""MMC simulation history data handling."""

from __future__ import annotations

# SPDX-License-Identifier: MIT

from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np


# MMC 2.8.0 (v2025.10, "Bubble Tea") writes a 64-byte MCXH header:
#
# magic,
# version, medianum, detnum, colcount, totalphoton, detected, savedphoton,
# unitinmm, seedbyte, normalizer, respin, srcnum, savedetflag,
# totalsource, reserved.
_HEADER = struct.Struct("<4s7IfIfi4I")

# savedetflag bits and field widths, in photon-record storage order.
_FIELD_SPECS = (
    ("detid", 0, lambda media: 1),
    ("nscat", 1, lambda media: media),
    ("ppath", 2, lambda media: media),
    ("mom", 3, lambda media: media),
    ("p", 4, lambda media: 3),
    ("v", 5, lambda media: 3),
    ("w0", 6, lambda media: 1),
    ("stokes", 7, lambda media: 4),
)

_REQUIRED_FIELDS = ("detid", "ppath")


def _read_exact(file: BinaryIO, size: int, description: str) -> bytes:
    """Read exactly ``size`` bytes."""
    data = file.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated MMC history file while reading {description}")
    return data


def _record_layout(
    medium_count: int,
    savedetflag: int,
) -> list[tuple[str, int]]:
    """Return enabled detected-photon fields and their column widths."""
    if savedetflag & ~0xFF:
        raise ValueError(f"Unsupported MMC savedetflag bits: 0x{savedetflag & ~0xFF:x}")

    return [(name, width_fn(medium_count)) for name, bit, width_fn in _FIELD_SPECS if savedetflag & (1 << bit)]


def read_flux(file_path: str | Path) -> np.ndarray:
    """Read flux values from an MMC ``.dat`` output file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the MMC ``.dat`` file.

    Returns
    -------
    numpy.ndarray
        Flux values from the second column.
    """
    flux_data = np.loadtxt(Path(file_path), ndmin=2)
    if flux_data.shape[1] < 2:
        raise ValueError("MMC flux output must contain at least two columns")
    return flux_data[:, 1]


def read_history(
    file_path: str | Path,
) -> dict[str, np.ndarray | float]:
    """Read detected-photon data from an MMC ``.mch`` file.

    This reader targets MMC 2.8.0 (v2025.10, "Bubble Tea").

    Photon path lengths are returned exactly as stored in the history file.
    The corresponding ``unitinmm`` scale is returned separately so that
    physical unit conversion is performed only when photon weights are
    computed.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the MMC ``.mch`` file.

    Returns
    -------
    dict
        Detected-photon data using MMC-style field names. ``detid``,
        ``ppath``, and ``unitinmm`` are always returned. Optional fields
        ``nscat``, ``mom``, ``p``, ``v``, ``w0``, and ``stokes`` are
        returned when enabled by ``savedetflag``.

    Raises
    ------
    ValueError
        If the file is malformed, uses an unsupported history version,
        contains inconsistent history blocks, or omits required fields.
    """
    path = Path(file_path)

    chunks: dict[str, list[np.ndarray]] = {}
    reference_layout: tuple[int, int, int, int, float] | None = None
    unitinmm_value: float | None = None
    block_count = 0

    with path.open("rb") as file:
        while True:
            first_four = file.read(4)
            if not first_four:
                break

            header_bytes = first_four + _read_exact(
                file,
                _HEADER.size - 4,
                "history header",
            )

            (
                magic,
                version,
                medium_count,
                detector_count,
                record_count,
                _total_photons,
                _detected_photons,
                saved_photons,
                unitinmm,
                seed_bytes,
                _normalizer,
                respin,
                source_count,
                savedetflag,
                _total_source,
                _reserved,
            ) = _HEADER.unpack(header_bytes)

            if magic != b"MCXH":
                raise ValueError("Invalid MMC history file: missing MCXH header")
            if version != 1:
                raise ValueError(f"Unsupported MMC history version {version}; expected version 1")

            layout = _record_layout(medium_count, savedetflag)
            expected_columns = sum(width for _, width in layout)
            if record_count != expected_columns:
                raise ValueError(
                    "MMC history record width does not match savedetflag: "
                    f"header reports {record_count} columns, "
                    f"savedetflag implies {expected_columns}"
                )

            block_layout = (
                medium_count,
                detector_count,
                record_count,
                savedetflag,
                float(unitinmm),
            )
            if reference_layout is None:
                reference_layout = block_layout
                unitinmm_value = float(unitinmm)
            elif block_layout != reference_layout:
                raise ValueError("MMC history contains incompatible history blocks")

            _ = respin, source_count

            num_values = saved_photons * record_count
            record_bytes = _read_exact(
                file,
                num_values * np.dtype("<f4").itemsize,
                "detected-photon records",
            )
            records = np.frombuffer(
                record_bytes,
                dtype="<f4",
            ).reshape(saved_photons, record_count)

            column = 0
            for name, width in layout:
                next_column = column + width
                values = records[:, column:next_column].copy()

                if width == 1:
                    values = values[:, 0]

                chunks.setdefault(name, []).append(values)
                column = next_column

            # Optional RNG seeds follow the detected-photon records.
            if seed_bytes:
                _read_exact(
                    file,
                    saved_photons * seed_bytes,
                    "photon RNG seeds",
                )

            block_count += 1

    if block_count == 0:
        raise ValueError("MMC history file contains no history blocks")

    missing = [name for name in _REQUIRED_FIELDS if name not in chunks]
    if missing:
        raise ValueError("MMC history output is missing required fields: " + ", ".join(missing))

    detected_photons: dict[str, np.ndarray | float] = {
        name: np.concatenate(parts, axis=0) for name, parts in chunks.items()
    }
    detected_photons["unitinmm"] = float(unitinmm_value)

    return detected_photons


def read_cli_output(
    file_stub: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Read flux and detected-photon output from an MMC CLI run.

    Parameters
    ----------
    file_stub : str or pathlib.Path
        Common path without the ``.dat`` or ``.mch`` suffix.

    Returns
    -------
    flux : numpy.ndarray
        Flux values from the second column of the ``.dat`` file.
    detected_photons : dict
        Detected-photon data parsed from the ``.mch`` file.
    """
    stub = Path(file_stub)
    flux = read_flux(stub.with_suffix(".dat"))
    detected_photons = read_history(stub.with_suffix(".mch"))
    return flux, detected_photons
