"""Resumable stage execution and human review gates shared by the packaged workflows.

A stage is reused only when its command (or action identity) and its recorded input identity both match the stage
marker and every required output exists; any mismatch raises instead of silently rerunning or adopting stale outputs.
Review gates publish a template file and stop with ``ReviewRequired`` until the author fills it in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


class ReviewRequired(RuntimeError):
    """Raised after publishing the next human-review gate."""

    def __init__(self, message: str, path: Path) -> None:
        super().__init__(f"{message}: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def gpu_uuids() -> tuple[str, ...]:
    """CUDA GPU UUIDs, largest memory first."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.total", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        uuid, memory = (piece.strip() for piece in line.split(",", 1))
        rows.append((int(memory), uuid))
    if not rows:
        raise RuntimeError("No CUDA GPUs reported by nvidia-smi")
    return tuple(uuid for _memory, uuid in sorted(rows, reverse=True))


def gate_numbers(
    path: Path,
    keys: Sequence[str],
    *,
    message: str,
    artifact_type: str,
    identity: dict[str, object] | None = None,
) -> dict[str, float]:
    """Return author-filled positive numbers for every key, or publish the template and stop.

    With ``identity``, the numbers are bound to the reviewed artifacts it names and refused once those change.
    """
    if not path.exists():
        template: dict[str, object] = {"artifact_type": artifact_type, "thresholds": {key: None for key in keys}}
        if identity is not None:
            template["identity"] = identity
        atomic_json(path, template)
        raise ReviewRequired(message, path)
    payload = json.loads(path.read_text())
    if identity is not None and payload.get("identity") != json.loads(json.dumps(identity)):
        raise ValueError(f"Numbers in {path} were read from different artifacts; delete it to review the current ones")
    values = payload.get("thresholds")
    if not isinstance(values, dict) or set(values) != set(keys):
        raise ReviewRequired(message, path)
    if any(not isinstance(values[key], (int, float)) or values[key] <= 0 for key in keys):
        raise ReviewRequired(message, path)
    return {key: float(values[key]) for key in keys}


def gate_choice(
    path: Path, options: dict[str, object], *, identity: dict[str, str], message: str, artifact_type: str
) -> str:
    """Return the author's selected option, bound to the measurements it was chosen from, or publish them and stop."""
    if not path.exists():
        atomic_json(path, {"artifact_type": artifact_type, "identity": identity, "options": options, "selected": None})
        raise ReviewRequired(message, path)
    payload = json.loads(path.read_text())
    if payload.get("identity") != identity:
        raise ValueError(f"Choice in {path} was made from different measurements; delete it to review the current ones")
    if payload.get("selected") not in options:
        raise ReviewRequired(message, path)
    return str(payload["selected"])


def gate_approval(path: Path, identity: dict[str, str], *, message: str, artifact_type: str) -> None:
    """Require approval bound to the exact reviewed artifacts."""
    expected = {"artifact_type": artifact_type, "approved": False, "identity": identity}
    if not path.exists():
        atomic_json(path, expected)
        raise ReviewRequired(message, path)
    payload = json.loads(path.read_text())
    if payload != expected | {"approved": True}:
        if payload.get("identity") != identity:
            raise ValueError(f"Approval identity no longer matches reviewed artifacts: {path}")
        raise ReviewRequired(message, path)


class StageRunner:
    def __init__(self, *, logs: Path, stages: Path, prefix: str, artifact_type: str, dry_run: bool) -> None:
        self.logs = logs
        self.stages = stages
        self.prefix = prefix
        self.artifact_type = artifact_type
        self.dry_run = dry_run
        logs.mkdir(parents=True, exist_ok=True)
        stages.mkdir(parents=True, exist_ok=True)

    def _marker(self, name: str) -> Path:
        return self.stages / f"{name}.json"

    def command(
        self,
        name: str,
        argv: Sequence[str | Path],
        *,
        required: Sequence[Path],
        env: dict[str, str] | None = None,
        validate: Callable[[], None] | None = None,
        input_identity: Callable[[], object] | None = None,
        attempts: int = 1,
    ) -> None:
        """Run one stage, or reuse its recorded completion.

        ``attempts`` above 1 reruns a command that exits nonzero, for commands that resume finished work and can die
        without an error of their own (Method 6, recover-native). It is process recovery, not a scientific retry.
        """
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}")
        command = [str(value) for value in argv]
        identity = json_sha256({"command": command, "environment": env or {}})
        marker = self._marker(name)
        inputs = input_identity() if input_identity is not None else None
        if marker.is_file():
            payload = json.loads(marker.read_text())
            if payload.get("identity") != identity:
                raise ValueError(f"Stage {name} command changed; refusing to reuse {marker}")
            if input_identity is not None and payload.get("inputs") != json.loads(json.dumps(inputs)):
                raise ValueError(f"Stage {name} inputs changed or were not recorded; refusing to reuse {marker}")
            if all(path.exists() for path in required):
                if validate is not None:
                    validate()
                print(f"[{self.prefix}] reuse {name}", flush=True)
                return
        print(f"[{self.prefix}] " + " ".join(command), flush=True)
        if self.dry_run:
            return
        # Record what is about to run before it runs: resumable commands reuse partial outputs, so an interrupted
        # stage may only resume under the same command and inputs.
        started = self.stages / f"{name}.started.json"
        launch = json.loads(json.dumps({"identity": identity, "inputs": inputs}))
        if started.is_file():
            previous = json.loads(started.read_text())
            if {key: previous.get(key) for key in launch} != launch:
                raise ValueError(
                    f"Stage {name} was interrupted under a different command or inputs ({started}); its partial "
                    f"outputs may not belong to this run. Remove them and {started} before rerunning"
                )
        else:
            atomic_json(
                started,
                {
                    "artifact_type": self.artifact_type,
                    "name": name,
                    **launch,
                    "command": command,
                    "started_at": datetime.now().astimezone().isoformat(),
                },
            )
        run_env = os.environ.copy()
        run_env.update(env or {})
        run_env.setdefault("PYTHONUNBUFFERED", "1")
        for attempt in range(1, attempts + 1):
            with (self.logs / f"{name}.log").open("a", buffering=1) as log:
                log.write(f"\n{datetime.now().astimezone().isoformat()} attempt {attempt} {' '.join(command)}\n")
                process = subprocess.Popen(
                    command, env=run_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                status = process.wait()
            if not status:
                break
            print(f"[{self.prefix}] {name} exited {status} on attempt {attempt} of {attempts}", flush=True)
        if status:
            raise subprocess.CalledProcessError(status, command)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Stage {name} completed without required outputs: {missing}")
        if validate is not None:
            validate()
        atomic_json(
            marker,
            {
                "artifact_type": self.artifact_type,
                "name": name,
                "identity": identity,
                "command": command,
                "inputs": inputs,
                "required_outputs": [str(path.resolve()) for path in required],
                "completed_at": datetime.now().astimezone().isoformat(),
            },
        )
        started.unlink()

    def action(
        self,
        name: str,
        identity_payload: object,
        action: Callable[[], None],
        *,
        required: Sequence[Path],
        validate: Callable[[], None] | None = None,
    ) -> None:
        identity = json_sha256(identity_payload)
        marker = self._marker(name)
        if marker.is_file() and json.loads(marker.read_text()).get("identity") == identity:
            if all(path.exists() for path in required):
                if validate is not None:
                    validate()
                print(f"[{self.prefix}] reuse {name}", flush=True)
                return
        print(f"[{self.prefix}] {name}", flush=True)
        if self.dry_run:
            return
        action()
        if validate is not None:
            validate()
        if not all(path.exists() for path in required):
            raise RuntimeError(f"Action {name} did not create all required outputs")
        atomic_json(
            marker,
            {
                "artifact_type": self.artifact_type,
                "name": name,
                "identity": identity,
                "completed_at": datetime.now().astimezone().isoformat(),
            },
        )
