from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from device_agent_arena.artifacts.run_io import artifact_tree_sha256
from device_agent_arena.v2.composition_models import V2CompositionTask
from device_agent_arena.v2.composition_reporting import build_composition_report
from device_agent_arena.v2.run_attestation import verify_model_run_attestation


def _verify_checkpoint(config: dict[str, Any]) -> dict[str, str]:
    candidate = config["candidate"]
    legacy = "checkpoint" in candidate
    path_value = str(
        candidate["checkpoint"] if legacy else candidate.get("model_path", "")
    )
    if not path_value:
        raise ValueError("composition gate config is missing candidate model path")
    checkpoint = Path(path_value)
    expected = str(
        candidate.get(
            "checkpoint_tree_sha256" if legacy else "model_tree_sha256", ""
        )
    )
    if not expected:
        raise ValueError("composition gate config is missing candidate model tree SHA-256")
    observed = artifact_tree_sha256(checkpoint)
    if observed != expected:
        raise ValueError("composition candidate checkpoint SHA-256 mismatch")
    binding = {"path": path_value, "sha256": observed}
    adapter_value = candidate.get("adapter_path")
    adapter_expected = str(candidate.get("adapter_sha256", ""))
    if (adapter_value is None) != (not adapter_expected):
        raise ValueError("composition gate adapter binding is incomplete")
    if adapter_value is not None:
        adapter_path = Path(str(adapter_value))
        weights_path = adapter_path / "adapter_model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError("composition candidate adapter weights are missing")
        adapter_observed = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if adapter_observed != adapter_expected:
            raise ValueError("composition candidate adapter SHA-256 mismatch")
        binding.update(
            {
                "adapter_path": str(adapter_value),
                "adapter_sha256": adapter_observed,
            }
        )
    return binding


def _candidate_agent_id(config: dict[str, Any]) -> str:
    candidate = config["candidate"]
    explicit = candidate.get("agent_id")
    if explicit:
        return str(explicit)
    return f"v2-composition-model:{candidate['checkpoint']}"


def _verify_report_run_binding(
    *,
    report: dict[str, Any],
    run_path: Path,
    checkpoint: dict[str, str],
    expected_runtime: dict[str, Any],
    expected_protocol: str,
) -> None:
    rebuilt = build_composition_report(run_path)
    if report != rebuilt:
        raise ValueError("composition report does not exactly match its raw run")
    verify_model_run_attestation(
        run_path,
        expected_protocol=expected_protocol,
        checkpoint=checkpoint,
        expected_runtime=expected_runtime,
    )


def _success_count(section: dict[str, Any], expected_count: int) -> int:
    if int(section.get("task_count", -1)) != expected_count:
        raise ValueError("composition report group count does not match frozen gate")
    return round(
        expected_count * float(section.get("safe_strict_success_rate", 0.0))
    )


def evaluate_composition_dev18_gate(
    *, config: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    expected_count = int(config["task_count"])
    expected_agent = _candidate_agent_id(config)
    expected_slices = {
        str(name): int(count) for name, count in config["slice_counts"].items()
    }
    if report.get("format_version") != "deviceops-ul-v2-composition-report-v0.1":
        raise ValueError("composition report format does not match frozen gate")
    if report.get("split") != "dev":
        raise ValueError("composition gate accepts only a Dev report")
    if int(report.get("task_count", -1)) != expected_count:
        raise ValueError("composition report task count does not match frozen gate")
    if report.get("task_suite_sha256") != config["task_suite_sha256"]:
        raise ValueError("composition report suite hash does not match frozen gate")
    if report.get("task_ids_sha256") != config["task_ids_sha256"]:
        raise ValueError("composition report task IDs do not match frozen gate")
    if report.get("agent_id") != expected_agent:
        raise ValueError("composition report does not belong to frozen candidate")
    by_slice = report.get("by_slice", {})
    if set(by_slice) != set(expected_slices):
        raise ValueError("composition report slices do not match frozen gate")

    thresholds = config["thresholds"]
    success_count = round(
        expected_count * float(report.get("safe_strict_success_rate", 0.0))
    )
    safe_count = round(
        expected_count * float(report.get("safe_execution_rate", 0.0))
    )
    minefield_count = round(
        expected_count * float(report.get("minefield_violation_rate", 0.0))
    )
    slice_counts = {
        name: _success_count(by_slice[name], count)
        for name, count in expected_slices.items()
    }
    recovery_task_count = int(
        thresholds["dual_user_failure_recovery_task_count"]
    )
    recovery_count = round(
        recovery_task_count
        * float(report.get("dual_user_failure_recovery_rate", 0.0))
    )
    binding_task_count = int(thresholds["cross_domain_output_binding_task_count"])
    binding_count = round(
        binding_task_count
        * float(report.get("cross_domain_output_binding_rate", 0.0))
    )
    infeasibility = report.get("infeasibility_detection", {})
    checks = {
        "safe_strict_success_count": success_count
        >= int(thresholds["safe_strict_success_count_min"]),
        "safe_execution_count": safe_count
        >= int(thresholds["safe_execution_count_min"]),
        "minefield_violation_count": minefield_count
        <= int(thresholds["minefield_violation_count_max"]),
        "mean_milestone_coverage": float(
            report.get("mean_milestone_coverage", 0.0)
        )
        >= float(thresholds["mean_milestone_coverage_min"]),
        "infeasibility_precision": float(infeasibility.get("precision", 0.0))
        >= float(thresholds["infeasibility_precision_min"]),
        "infeasibility_recall": float(infeasibility.get("recall", 0.0))
        >= float(thresholds["infeasibility_recall_min"]),
        "infeasibility_true_negative_rate": float(
            infeasibility.get("true_negative_rate", 0.0)
        )
        >= float(thresholds["infeasibility_true_negative_rate_min"]),
        "dual_user_failure_recovery_count": recovery_count
        >= int(thresholds["dual_user_failure_recovery_count_min"]),
        "cross_domain_output_binding_count": binding_count
        >= int(thresholds["cross_domain_output_binding_count_min"]),
    }
    for name, count in slice_counts.items():
        checks[f"slice.{name}_success_count"] = count >= int(
            thresholds["slice_success_count_min"][name]
        )

    return {
        "format_version": "deviceops-ul-v2-composition-dev18-receipt-v0.4",
        "status": "pass" if all(checks.values()) else "fail",
        "candidate_agent_id": expected_agent,
        "task_count": expected_count,
        "task_suite_sha256": config["task_suite_sha256"],
        "task_ids_sha256": config["task_ids_sha256"],
        "slice_counts": expected_slices,
        "checks": checks,
        "observed": {
            "safe_strict_success_count": success_count,
            "safe_execution_count": safe_count,
            "minefield_violation_count": minefield_count,
            "mean_milestone_coverage": float(
                report.get("mean_milestone_coverage", 0.0)
            ),
            "slice_success_counts": slice_counts,
            "infeasibility_detection": infeasibility,
            "dual_user_failure_recovery_count": recovery_count,
            "cross_domain_output_binding_count": binding_count,
        },
        "thresholds": thresholds,
    }


def write_composition_dev18_receipt(
    *,
    config_path: str | Path,
    report_path: str | Path,
    run_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    inputs = {
        "config": Path(config_path),
        "report": Path(report_path),
        "run": Path(run_path),
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in inputs.items()
        if name != "run"
    }
    checkpoint = _verify_checkpoint(payloads["config"])
    _verify_report_run_binding(
        report=payloads["report"],
        run_path=inputs["run"],
        checkpoint=checkpoint,
        expected_runtime=payloads["config"]["candidate"]["runtime"],
        expected_protocol=str(
            payloads["config"]["candidate"].get(
                "protocol", "deviceops-ul-v2-composition-public"
            )
        ),
    )
    receipt = evaluate_composition_dev18_gate(
        config=payloads["config"], report=payloads["report"]
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
        raise FileExistsError(
            f"refusing to overwrite composition promotion receipt: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def verify_composition_dev18_receipt(
    receipt_path: str | Path, *, candidate_agent_id: str
) -> dict[str, Any]:
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.get("status") != "pass":
        raise ValueError("composition promotion receipt did not pass; full Dev remains locked")
    if receipt.get("candidate_agent_id") != candidate_agent_id:
        raise ValueError("composition receipt candidate does not match requested model")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"config", "report", "run"}:
        raise ValueError("composition receipt authenticated input set is incomplete")
    payloads: dict[str, dict[str, Any]] = {}
    for name in ("config", "report", "run"):
        record = inputs[name]
        path = Path(record["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
            raise ValueError(f"composition receipt {name} SHA-256 mismatch")
        if name != "run":
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = _verify_checkpoint(payloads["config"])
    _verify_report_run_binding(
        report=payloads["report"],
        run_path=Path(inputs["run"]["path"]),
        checkpoint=checkpoint,
        expected_runtime=payloads["config"]["candidate"]["runtime"],
        expected_protocol=str(
            payloads["config"]["candidate"].get(
                "protocol", "deviceops-ul-v2-composition-public"
            )
        ),
    )
    if receipt.get("candidate_checkpoint") != checkpoint:
        raise ValueError("composition receipt checkpoint binding no longer verifies")
    recomputed = evaluate_composition_dev18_gate(
        config=payloads["config"], report=payloads["report"]
    )
    for key in (
        "format_version",
        "status",
        "candidate_agent_id",
        "task_count",
        "task_suite_sha256",
        "task_ids_sha256",
        "slice_counts",
        "checks",
        "observed",
        "thresholds",
    ):
        if receipt.get(key) != recomputed.get(key):
            raise ValueError(f"composition receipt field no longer verifies: {key}")
    return receipt


def verify_composition_dev18_task_slice(
    receipt: dict[str, Any], tasks: Sequence[V2CompositionTask]
) -> None:
    count = int(receipt["task_count"])
    if len(tasks) < count:
        raise ValueError("composition Dev suite is smaller than authorized gate slice")
    gate_tasks = tuple(tasks[:count])
    payload = (
        "".join(
            json.dumps(asdict(task), ensure_ascii=False, sort_keys=True) + "\n"
            for task in gate_tasks
        )
    ).encode("utf-8")
    suite_sha = hashlib.sha256(payload).hexdigest()
    ids_sha = hashlib.sha256(
        ("\n".join(task.id for task in gate_tasks) + "\n").encode("utf-8")
    ).hexdigest()
    if suite_sha != receipt.get("task_suite_sha256"):
        raise ValueError("composition Dev suite does not match authorized gate slice")
    if ids_sha != receipt.get("task_ids_sha256"):
        raise ValueError("composition Dev task IDs do not match authorized gate slice")
