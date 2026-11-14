import subprocess
from pathlib import Path

import pytest

from mmcnirs.mmc import runner


@pytest.mark.parametrize(("version", "update_manifest"), [(None, False), ("v2025.10", False), ("v2026", True)])
def test_run_mmc_executes_managed_runtime_and_returns_output(
    tmp_path: Path, monkeypatch, version, update_manifest
) -> None:
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

    def fake_get_mmc_executable(**kwargs) -> Path:
        runtime_calls.append(kwargs)
        return executable

    def fake_run(command, **kwargs):
        subprocess_calls.append((command, kwargs))
        return completed_process

    monkeypatch.setattr(runner.runtime, "get_mmc_executable", fake_get_mmc_executable)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    result = runner.run_mmc(
        "simulation.json", working_directory=tmp_path, timeout=42, version=version, update_manifest=update_manifest
    )

    assert result is completed_process
    assert result.stdout == "simulation complete"
    assert runtime_calls == [{"version": version, "update_manifest": update_manifest}]
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


def test_run_mmc_raises_clear_timeout_after_max_trials(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "simulation.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner.runtime, "get_mmc_executable", lambda **kwargs: tmp_path / "runtime" / "mmc")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], stderr="MMC stalled on detector 16")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(
        TimeoutError,
        match="MMC timed out after 3 seconds.*simulation.json.*Command.*MMC stalled on detector 16",
    ):
        runner.run_mmc(config_path, working_directory=tmp_path, timeout=3)

    assert len(calls) == 5


def test_run_mmc_raises_clear_error_for_nonzero_exit_without_retry(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "simulation.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner.runtime, "get_mmc_executable", lambda **kwargs: tmp_path / "runtime" / "mmc")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode=7, stdout="partial output", stderr="invalid mesh")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="MMC exited with code 7.*invalid mesh"):
        runner.run_mmc(config_path, working_directory=tmp_path)

    assert len(calls) == 1


def test_run_mmc_uses_pre_resolved_executable_without_runtime_lookup(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "simulation.json"
    config_path.write_text("{}", encoding="utf-8")
    executable = tmp_path / "runtime/v2026/mmc"
    monkeypatch.setattr(
        runner.runtime, "get_mmc_executable", lambda **kwargs: pytest.fail("Runtime was resolved again")
    )

    def fake_run(command, **kwargs):
        assert command[0] == str(executable.resolve())
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner.run_mmc(config_path, working_directory=tmp_path, executable=executable)
