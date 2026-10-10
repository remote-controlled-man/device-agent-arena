from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from device_agent_arena.v1.reporting import build_v1_report, load_v1_run

_SCHEMA_PROFILES = {"identity", "schema-a", "schema-b"}


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _trial_profile(trial: dict[str, Any], fallback: str) -> str:
    candidate = str(trial.get("agent_name", "")).rsplit(":", 1)[-1]
    return candidate if candidate in _SCHEMA_PROFILES else fallback


def build_v2_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    header, trials = load_v1_run(source)
    report = build_v1_report(source)
    configured = str(header.get("agent_config", {}).get("schema_profile", "unknown"))
    fallback = configured if configured in _SCHEMA_PROFILES else "unknown"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[_trial_profile(trial, fallback)].append(trial)
    report["format_version"] = "deviceops-ul-report-v2.0"
    report["run_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["task_ids_sha256"] = header["task_ids_sha256"]
    report["condition"] = {
        "schema_profile": configured,
        "shared_impact_guard": bool(
            header.get("agent_config", {}).get("shared_impact_guard", False)
        ),
    }
    report["by_schema_profile"] = {
        profile: {
            "task_count": len(items),
            "safe_strict_success_rate": _rate(
                [bool(item["grade"]["success"]) for item in items]
            ),
            "goal_completion_rate": _rate(
                [bool(item["grade"]["goal_completion"]) for item in items]
            ),
            "safe_execution_rate": _rate(
                [bool(item["grade"]["safe_execution"]) for item in items]
            ),
            "boundary_violation_rate": _rate(
                [bool(item["grade"]["violations"]) for item in items]
            ),
        }
        for profile, items in sorted(grouped.items())
    }
    return report


def write_v2_report(run_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    report = build_v2_report(run_path)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite V2 report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
