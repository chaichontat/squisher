from __future__ import annotations

from squisher_lightsheet.artifacts import stamp_artifact


def test_stamp_artifact_sets_current_stage_type() -> None:
    stamped = stamp_artifact(
        {"schema_version": 1, "artifact_type": "stale"},
        "lightsheet.position.v1",
    )

    assert stamped["schema_version"] == 1
    assert stamped["artifact_type"] == "lightsheet.position.v1"
