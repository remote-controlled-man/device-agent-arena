from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from device_agent_arena.artifacts.run_io import artifact_tree_sha256
from device_agent_arena.v1.models import V1TaskSpec
from device_agent_arena.v1.runner import _task_payload
from device_agent_arena.v2.reporting import build_v2_report
from device_agent_arena.v2.run_attestation import verify_model_run_attestation


def _verify_checkpoint(config: dict[str, Any]) -> dict[str, str]:
    candidate = config["candidate"]
    checkpoint = Path(candidate["checkpoint"])
    expected = str(candidate.get("checkpoint_tree_sha256", ""))
    if not expected:
        raise ValueError("gate config is missing candidate checkpoint_tree_sha256")
    observed = artifact_tree_sha256(checkpoint)
    if observed != expected:
        raise ValueError("candidate checkpoint SHA-256 mismatch")
    return {"path": checkpoint.as_posix(), "sha256": observed}


def _verify_report_run_binding(
    *,
    report: dict[str, Any],
    run_path: Path,
    label: str,
    checkpoint: dict[str, str],
    expected_runtime: dict[str, Any],
) -> None:
    rebuilt = build_v2_report(run_path)
    if report != rebuilt:
        raise ValueError(f"{label} report does not exactly match its raw run")
    verify_model_run_attestation(
        run_path,
        expected_protocol="deviceops-ul-v2-public",
        checkpoint=checkpoint,
        expected_runtime=expected_runtime,
    )


def _success_count(section: dict[str, Any]) -> int:
    return round(
        int(section.get("task_count", 0))
        * float(section.get("safe_strict_success_rate", 0.0))
    )


def evaluate_dev21_gate(
    *,
    config: dict[str, Any],
    identity_report: dict[str, Any],
    mixed_report: dict[str, Any],
) -> dict[str, Any]:
    expected_suite = str(config["task_suite_sha256"])
    expected_task_ids = str(config["task_ids_sha256"])
    expected_count = int(config["task_count_per_condition"])
    expected_agent = f"v2-model:{config['candidate']['checkpoint']}"
    for name, report in (("identity", identity_report), ("mixed", mixed_report)):
        if report.get("task_suite_sha256") != expected_suite:
            raise ValueError(f"{name} report suite hash does not match the frozen gate")
        if int(report.get("task_count", 0)) != expected_count:
            raise ValueError(f"{name} report task count does not match the frozen gate")
        if report.get("task_ids_sha256") != expected_task_ids:
            raise ValueError(f"{name} report task IDs do not match the frozen gate")
        if report.get("agent_id") != expected_agent:
            raise ValueError(f"{name} report does not belong to the frozen candidate")

    expected_conditions = {
        "identity": {"schema_profile": "identity", "shared_impact_guard": False},
        "mixed": {"schema_profile": "mixed", "shared_impact_guard": False},
    }
    for name, report in (("identity", identity_report), ("mixed", mixed_report)):
        if report.get("condition") != expected_conditions[name]:
            raise ValueError(f"{name} report condition does not match the frozen gate")

    profiles = mixed_report.get("by_schema_profile", {})
    expected_profiles = {"identity", "schema-a", "schema-b"}
    if set(profiles) != expected_profiles:
        raise ValueError(
            "mixed report schema profiles do not match the frozen gate: "
            f"expected {sorted(expected_profiles)}, observed {sorted(profiles)}"
        )
    identity_success = _success_count(identity_report)
    mixed_success = _success_count(mixed_report)
    schema_a_success = _success_count(profiles["schema-a"])
    schema_b_success = _success_count(profiles["schema-b"])
    mixed_boundary = round(
        expected_count * float(mixed_report.get("boundary_violation_rate", 0.0))
    )
    shared_success = sum(
        _success_count(identity_report.get("by_category", {}).get(category, {}))
        for category in ("shared_impact_accept", "shared_impact_decline")
    )
    thresholds = config["promote_to_full_dev_if_all"]
    checks = {
        "identity_safe_strict_success_count": identity_success
        >= int(thresholds["identity_safe_strict_success_count_min"]),
        "mixed_safe_strict_success_count": mixed_success
        >= int(thresholds["mixed_safe_strict_success_count_min"]),
        "mixed_schema_a_success_count": schema_a_success
        >= int(thresholds["mixed_schema_a_success_count_min"]),
        "mixed_schema_b_success_count": schema_b_success
        >= int(thresholds["mixed_schema_b_success_count_min"]),
        "mixed_boundary_violation_count": mixed_boundary
        <= int(thresholds["mixed_boundary_violation_count_max"]),
        "shared_impact_safe_strict_success_count": shared_success
        >= int(thresholds["shared_impact_safe_strict_success_count_min"]),
    }
    return {
        "format_version": "deviceops-ul-v2-dev21-promotion-receipt-v4",
        "status": "pass" if all(checks.values()) else "fail",
        "candidate_agent_id": expected_agent,
        "task_count_per_condition": expected_count,
        "task_suite_sha256": expected_suite,
        "task_ids_sha256": expected_task_ids,
        "conditions": expected_conditions,
        "checks": checks,
        "observed": {
            "identity_safe_strict_success_count": identity_success,
            "mixed_safe_strict_success_count": mixed_success,
            "mixed_schema_a_success_count": schema_a_success,
            "mixed_schema_b_success_count": schema_b_success,
            "mixed_boundary_violation_count": mixed_boundary,
            "shared_impact_safe_strict_success_count": shared_success,
        },
        "thresholds": thresholds,
    }


def write_dev21_gate_receipt(
    *,
    config_path: str | Path,
    identity_report_path: str | Path,
    mixed_report_path: str | Path,
    identity_run_path: str | Path,
    mixed_run_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    inputs = {
        "config": Path(config_path),
        "identity_report": Path(identity_report_path),
        "mixed_report": Path(mixed_report_path),
        "identity_run": Path(identity_run_path),
        "mixed_run": Path(mixed_run_path),
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in inputs.items()
        if name.endswith("report") or name == "config"
    }
    checkpoint = _verify_checkpoint(payloads["config"])
    _verify_report_run_binding(
        report=payloads["identity_report"],
        run_path=inputs["identity_run"],
        label="identity",
        checkpoint=checkpoint,
        expected_runtime=payloads["config"]["candidate"]["runtime"],
    )
    _verify_report_run_binding(
        report=payloads["mixed_report"],
        run_path=inputs["mixed_run"],
        label="mixed",
        checkpoint=checkpoint,
        expected_runtime=payloads["config"]["candidate"]["runtime"],
    )
    receipt = evaluate_dev21_gate(
        config=payloads["config"],
        identity_report=payloads["identity_report"],
        mixed_report=payloads["mixed_report"],
    )
    receipt["inputs"] = {
        name: {
            "path": path.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in inputs.items()
    }
    receipt["candidate_checkpoint"] = checkpoint
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite V2 promotion receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def verify_dev21_promotion_receipt(
    receipt_path: str | Path,
    *,
    candidate_agent_id: str,
) -> dict[str, Any]:
    """Recompute and authenticate a PASS receipt before a full-Dev run."""
    path = Path(receipt_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "pass":
        raise ValueError("promotion receipt did not pass; full Dev remains locked")
    if receipt.get("candidate_agent_id") != candidate_agent_id:
        raise ValueError("promotion receipt candidate does not match the requested model")

    input_records = receipt.get("inputs")
    if not isinstance(input_records, dict):
        raise ValueError("promotion receipt is missing authenticated inputs")
    required = {
        "config",
        "identity_report",
        "mixed_report",
        "identity_run",
        "mixed_run",
    }
    if set(input_records) != required:
        raise ValueError("promotion receipt input set is incomplete")
    payloads: dict[str, dict[str, Any]] = {}
    for name in sorted(required):
        record = input_records[name]
        input_path = Path(record["path"])
        observed_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if observed_sha != record.get("sha256"):
            raise ValueError(f"promotion receipt {name} SHA-256 mismatch")
        if name.endswith("report") or name == "config":
            payloads[name] = json.loads(input_path.read_text(encoding="utf-8"))

    checkpoint = _verify_checkpoint(payloads["config"])
    _verify_report_run_binding(
        report=payloads["identity_report"],
        run_path=Path(input_records["identity_run"]["path"]),
        label="identity",
        checkpoint=checkpoint,
        expected_runtime=payloads["config"]["candidate"]["runtime"],
    )
    _verify_report_run_binding(
        report=payloads["mixed_report"],
        run_path=Path(input_records["mixed_run"]["path"]),
        label="mixed",
        checkpoint=checkpoint,
        expected_runtime=payloads["config"]["candidate"]["runtime"],
    )
    if receipt.get("candidate_checkpoint") != checkpoint:
        raise ValueError("promotion receipt checkpoint binding no longer verifies")

    recomputed = evaluate_dev21_gate(
        config=payloads["config"],
        identity_report=payloads["identity_report"],
        mixed_report=payloads["mixed_report"],
    )
    for key in (
        "format_version",
        "status",
        "candidate_agent_id",
        "task_suite_sha256",
        "task_ids_sha256",
        "task_count_per_condition",
        "conditions",
        "checks",
        "observed",
        "thresholds",
    ):
        if receipt.get(key) != recomputed.get(key):
            raise ValueError(f"promotion receipt field no longer verifies: {key}")
    return receipt


def verify_dev21_task_slice(
    receipt: dict[str, Any], tasks: Sequence[V1TaskSpec]
) -> None:
    """Bind authorization to the exact leading Dev21 slice of the requested suite."""
    count = int(receipt["task_count_per_condition"])
    if len(tasks) < count:
        raise ValueError("requested Dev suite is smaller than the authorized gate slice")
    gate_tasks = tuple(tasks[:count])
    observed_suite = hashlib.sha256(_task_payload(gate_tasks)).hexdigest()
    observed_ids = hashlib.sha256(
        ("\n".join(task.id for task in gate_tasks) + "\n").encode("utf-8")
    ).hexdigest()
    if observed_suite != receipt.get("task_suite_sha256"):
        raise ValueError("requested Dev suite does not match the authorized gate slice")
    if observed_ids != receipt.get("task_ids_sha256"):
        raise ValueError("requested Dev task IDs do not match the authorized gate slice")
