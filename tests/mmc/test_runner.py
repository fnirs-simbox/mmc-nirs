import subprocess
from pathlib import Path

import pytest

from mmc_nirs.mmc import runner


def test_run_mmc_executes_managed_runtime_and_returns_output(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "simulation.json"
    config_path.write_text("{}", encoding="utf-8")
    executable = tmp_path / "runtime" / "mmc"
    completed_process = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="simulation complete",
        stderr="",
    )
    runtime_calls = []
    subprocess_calls = []

    def fake_get_mmc_executable() -> Path:
        runtime_calls.append(None)
        return executable

    def fake_run(command, **kwargs):
        subprocess_calls.append((command, kwargs))
        return completed_process

    monkeypatch.setattr(runner.runtime, "get_mmc_executable", fake_get_mmc_executable)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    result = runner.run_mmc("simulation.json", working_directory=tmp_path, timeout=42)

    assert result is completed_process
    assert result.stdout == "simulation complete"
    assert runtime_calls == [None]
    assert subprocess_calls == [
        (
            [str(executable.resolve()), "-f", str(config_path.resolve()), "-d", "1"],
            {
                "cwd": tmp_path.resolve(),
                "timeout": 42,
                "capture_output": True,
                "text": True,
                "check": False,
            },
        )
    ]
    assert Path(subprocess_calls[0][0][0]).is_absolute()


def test_run_mmc_rejects_missing_config_before_runtime_lookup(tmp_path: Path, monkeypatch) -> None:
    def unexpected_runtime_lookup() -> Path:
        pytest.fail("A missing config should be rejected before resolving the MMC runtime")

    monkeypatch.setattr(runner.runtime, "get_mmc_executable", unexpected_runtime_lookup)

    with pytest.raises(FileNotFoundError, match="MMC configuration file not found.*missing.json"):
        runner.run_mmc("missing.json", working_directory=tmp_path)


def test_run_mmc_raises_clear_timeout_without_retry(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "simulation.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner.runtime, "get_mmc_executable", lambda: tmp_path / "runtime" / "mmc")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(TimeoutError, match="MMC timed out after 3 seconds.*simulation.json"):
        runner.run_mmc(config_path, working_directory=tmp_path, timeout=3)

    assert len(calls) == 1


def test_run_mmc_raises_clear_error_for_nonzero_exit_without_retry(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "simulation.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner.runtime, "get_mmc_executable", lambda: tmp_path / "runtime" / "mmc")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode=7, stdout="partial output", stderr="invalid mesh")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="MMC exited with code 7.*invalid mesh"):
        runner.run_mmc(config_path, working_directory=tmp_path)

    assert len(calls) == 1
