"""Subprocess execution boundary for the external MMC runtime."""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
from pathlib import Path

from mmc_nirs.mmc import runtime

__all__ = ["run_mmc"]


def run_mmc(
    config_path: str | Path,
    *,
    working_directory: str | Path,
    timeout: float = 900,
) -> subprocess.CompletedProcess[str]:
    """Run MMC once for a configuration and return the completed process.

    Relative configuration paths are interpreted relative to
    ``working_directory``. Standard output and standard error are captured as
    text and included in execution errors when available.
    """
    resolved_working_directory = Path(working_directory).expanduser().resolve()
    resolved_config_path = Path(config_path).expanduser()
    if not resolved_config_path.is_absolute():
        resolved_config_path = resolved_working_directory / resolved_config_path
    resolved_config_path = resolved_config_path.resolve()

    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"MMC configuration file not found: {resolved_config_path}")

    executable = runtime.get_mmc_executable().resolve()
    command = [str(executable), "-f", str(resolved_config_path), "-d", "1"]
    try:
        completed_process = subprocess.run(
            command,
            cwd=resolved_working_directory,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"MMC timed out after {timeout} seconds while running {resolved_config_path}") from error

    if completed_process.returncode != 0:
        message = f"MMC exited with code {completed_process.returncode} while running {resolved_config_path}"
        diagnostics = completed_process.stderr.strip() or completed_process.stdout.strip()
        if diagnostics:
            message = f"{message}: {diagnostics}"
        raise RuntimeError(message)

    return completed_process
