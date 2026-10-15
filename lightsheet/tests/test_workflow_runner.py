from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from squisher_lightsheet.workflow_runner import StageRunner


def _runner(tmp_path: Path) -> StageRunner:
    return StageRunner(
        logs=tmp_path / "logs", stages=tmp_path / "stages", prefix="t", artifact_type="t.v1", dry_run=False
    )


def test_a_resumable_command_that_dies_is_rerun_within_its_attempts(tmp_path: Path) -> None:
    marker, output = tmp_path / "died-once", tmp_path / "out"
    script = f"test -e {marker} || {{ touch {marker}; exit 1; }}; touch {output}"

    _runner(tmp_path).command("stage", ["sh", "-c", script], required=[output], attempts=2)

    assert output.exists()
    assert " attempt 2 sh -c " in (tmp_path / "logs" / "stage.log").read_text()


def test_a_command_is_not_rerun_by_default(tmp_path: Path) -> None:
    marker, output = tmp_path / "died-once", tmp_path / "out"
    script = f"test -e {marker} || {{ touch {marker}; exit 1; }}; touch {output}"

    with pytest.raises(subprocess.CalledProcessError):
        _runner(tmp_path).command("stage", ["sh", "-c", script], required=[output])
    assert not output.exists()


def test_an_interrupted_stage_only_resumes_under_the_same_inputs(tmp_path: Path) -> None:
    source, ready, output = tmp_path / "source", tmp_path / "ready", tmp_path / "out"
    source.write_text("first")
    runner = _runner(tmp_path)
    argv = ["sh", "-c", f"test -e {ready} && touch {output}"]

    def run() -> None:
        runner.command("stage", argv, required=[output], input_identity=lambda: source.read_text())

    with pytest.raises(subprocess.CalledProcessError):
        run()
    ready.touch()
    source.write_text("second")
    with pytest.raises(ValueError, match="interrupted under a different command or inputs"):
        run()

    source.write_text("first")
    run()
    assert output.exists() and not (tmp_path / "stages" / "stage.started.json").exists()
