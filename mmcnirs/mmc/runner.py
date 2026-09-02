"""Subprocess execution boundary for the external MMC runtime."""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
from pathlib import Path

from mmcnirs.mmc import runtime

__all__ = ["run_mmc"]


def run_mmc(
    config_path: str | Path,
    *,
    working_directory: str | Path,
    timeout: float = 180,
    max_trials: int = 5,
) -> subprocess.CompletedProcess[str]:
    """Run MMC for a configuration and return the completed process.

    Relative configuration paths are interpreted relative to
    ``working_directory``. Standard output and standard error are captured as
    text and included in execution errors when available. Timed-out runs are
    retried up to ``max_trials`` total attempts.
    """
    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")

    resolved_working_directory = Path(working_directory).expanduser().resolve()
    resolved_config_path = Path(config_path).expanduser()
    if not resolved_config_path.is_absolute():
        resolved_config_path = resolved_working_directory / resolved_config_path
    resolved_config_path = resolved_config_path.resolve()

    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"MMC configuration file not found: {resolved_config_path}")

    executable = runtime.get_mmc_executable().resolve()
    command = [str(executable), "-f", str(resolved_config_path), "-d", "1"]
    for trial in range(1, max_trials + 1):
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
            if trial == max_trials:
                message = (
                    f"MMC timed out after {timeout} seconds for all {max_trials} trials "
                    f"while running {resolved_config_path}. Last subprocess error: {error}"
                )
                diagnostics = error.stderr or error.stdout
                if diagnostics:
                    if isinstance(diagnostics, bytes):
                        diagnostics = diagnostics.decode(errors="replace")
                    message = f"{message}: {diagnostics.strip()}"
                raise TimeoutError(message) from error
            continue

        if completed_process.returncode != 0:
            message = f"MMC exited with code {completed_process.returncode} while running {resolved_config_path}"
            diagnostics = completed_process.stderr.strip() or completed_process.stdout.strip()
            if diagnostics:
                message = f"{message}: {diagnostics}"
            raise RuntimeError(message)

        return completed_process

    raise AssertionError("unreachable")
