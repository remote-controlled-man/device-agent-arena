from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record_type") != "run_header":
        raise ValueError("V1 run must start with a run_header")
    trials = [record for record in records[1:] if record.get("record_type") == "trial"]
    if int(records[0].get("task_count", len(trials))) != len(trials):
        raise ValueError("V1 run header task_count does not match trial records")
    return records[0], trials


def _tool_result_codes(trial: dict[str, Any]) -> list[str]:
    return [
        str(event.get("payload", {}).get("code"))
        for event in trial.get("events", [])
        if event.get("kind") == "tool_result"
    ]


def _has_failed_replay(trial: dict[str, Any]) -> bool:
    definitive_failures: set[str] = set()
    for event in trial.get("events", []):
        if event.get("kind") != "tool_result":
            continue
        payload = event.get("payload", {})
        data = payload.get("data", {})
        if not isinstance(data, dict):
            continue
        operation_id = data.get("operation_id")
        if not isinstance(operation_id, str):
            continue
        if data.get("status") == "failed" or payload.get("code") in {
            "precondition_failed",
            "invalid_arguments",
            "permission_denied",
        }:
            definitive_failures.add(operation_id)
        if (
            payload.get("code") == "idempotent_replay"
            and data.get("status") == "failed"
            and operation_id in definitive_failures
        ):
            return True
    return False


def _root_cause(trial: dict[str, Any], archetype: str) -> str:
    codes = set(_tool_result_codes(trial))
    violations = set(trial.get("grade", {}).get("violations", []))
    if _has_failed_replay(trial):
        return "idempotency_key_reused_after_definitive_failure"
    if archetype.startswith("shared_impact") and (
        "action_before_confirmation" in violations
        or any(str(item).startswith(("wrong_target", "overscope")) for item in violations)
    ):
        return "shared_impact_order_or_target"
    if {"invalid_arguments", "unknown_tool"} & codes:
        return "contract_or_dialogue_format"
    return "model_policy_or_completion"


def analyze_v1_run(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    header, trials = _load_run(source)
    failures = [trial for trial in trials if not trial.get("grade", {}).get("success", False)]
    archetypes: Counter[str] = Counter()
    root_causes: Counter[str] = Counter()
    root_cause_task_ids: defaultdict[str, list[str]] = defaultdict(list)
    failed_checks: Counter[str] = Counter()
    tool_result_codes: Counter[str] = Counter()
    for trial in failures:
        categories = [str(item) for item in trial.get("categories", [])]
        archetype = next(
            (item for item in categories if item != "partial_contract"),
            "uncategorized",
        )
        archetypes[archetype] += 1
        cause = _root_cause(trial, archetype)
        root_causes[cause] += 1
        root_cause_task_ids[cause].append(str(trial.get("task_id", "unknown")))
        failed_checks.update(
            name
            for name, passed in trial.get("grade", {}).get("checks", {}).items()
            if not passed
        )
        tool_result_codes.update(_tool_result_codes(trial))
    trial_count = len(trials)
    failure_count = len(failures)
    return {
        "format_version": "deviceops-ul-v1-failure-analysis-v1.0",
        "input": source.as_posix(),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "agent_id": header.get("agent_id"),
        "split": header.get("split"),
        "trial_count": trial_count,
        "success_count": trial_count - failure_count,
        "failure_count": failure_count,
        "success_rate": (trial_count - failure_count) / trial_count if trial_count else 0.0,
        "failures_by_archetype": dict(sorted(archetypes.items())),
        "root_causes": dict(sorted(root_causes.items())),
        "root_cause_task_ids": {
            key: sorted(values) for key, values in sorted(root_cause_task_ids.items())
        },
        "failed_checks": dict(sorted(failed_checks.items())),
        "failed_trial_tool_result_codes": dict(sorted(tool_result_codes.items())),
        "method": (
            "Deterministic trajectory signatures over raw failed trials; root-cause "
            "labels are diagnostic and do not alter benchmark grades."
        ),
    }


def write_failure_analysis(path: str | Path, output: str | Path) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite failure analysis: {destination}")
    result = analyze_v1_run(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result
