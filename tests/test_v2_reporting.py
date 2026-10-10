from __future__ import annotations

import json
from pathlib import Path

from device_agent_arena.v2.reporting import build_v2_report


def _trial(profile: str, *, success: bool) -> dict:
    return {
        "record_type": "trial",
        "agent_name": f"mutation-coordinator:v2-model:test:{profile}",
        "task_id": f"task-{profile}",
        "categories": ["test"],
        "events": [],
        "fault_planned": False,
        "termination": "agent_finished",
        "required_clarifications": [],
        "required_confirmations": [],
        "grade": {
            "success": success,
            "goal_completion": success,
            "safe_execution": True,
            "violations": [],
            "checks": {},
            "diagnostics": {
                "clarification_count": 0,
                "confirmation_count": 0,
                "unnecessary_interactions": 0,
                "total_cost": 1.0,
            },
        },
    }


def test_v2_report_breaks_out_schema_profiles(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    records = [
        {
            "record_type": "run_header",
            "agent_id": "test",
            "split": "dev",
            "task_count": 3,
            "task_suite_sha256": "suite",
            "task_ids_sha256": "task-ids",
            "agent_config": {"schema_profile": "mixed"},
        },
        _trial("identity", success=True),
        _trial("schema-a", success=False),
        _trial("schema-b", success=True),
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    report = build_v2_report(path)

    assert report["format_version"] == "deviceops-ul-report-v2.0"
    assert report["task_ids_sha256"] == "task-ids"
    assert report["by_schema_profile"] == {
        "identity": {
            "boundary_violation_rate": 0.0,
            "goal_completion_rate": 1.0,
            "safe_execution_rate": 1.0,
            "safe_strict_success_rate": 1.0,
            "task_count": 1,
        },
        "schema-a": {
            "boundary_violation_rate": 0.0,
            "goal_completion_rate": 0.0,
            "safe_execution_rate": 1.0,
            "safe_strict_success_rate": 0.0,
            "task_count": 1,
        },
        "schema-b": {
            "boundary_violation_rate": 0.0,
            "goal_completion_rate": 1.0,
            "safe_execution_rate": 1.0,
            "safe_strict_success_rate": 1.0,
            "task_count": 1,
        },
    }
