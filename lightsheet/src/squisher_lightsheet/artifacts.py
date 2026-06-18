from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def stamp_artifact(payload: dict[str, Any], artifact_type: str) -> dict[str, Any]:
    updated = dict(payload)
    updated["schema_version"] = SCHEMA_VERSION
    updated["artifact_type"] = artifact_type
    return updated


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_workflow_summary(
    path: Path,
    *,
    workflow: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    parameters: dict[str, Any],
    commands: dict[str, str],
) -> None:
    write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "lightsheet.workflow_summary.v1",
            "workflow": workflow,
            "created_at": datetime.now(UTC).isoformat(),
            "inputs": inputs,
            "outputs": outputs,
            "parameters": parameters,
            "commands": commands,
        },
    )
