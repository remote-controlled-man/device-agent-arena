from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from device_agent_arena.v2.composition_reporting import (
    build_composition_report,
    load_composition_run,
)

_PLANNER_FIELDS = (
    "model_id",
    "model_artifact_sha256",
    "adapter_path",
    "adapter_sha256",
    "temperature",
    "max_new_tokens",
    "load_in_4bit",
    "chat_template_sha256",
)


def _planner_config(header: dict[str, Any]) -> dict[str, Any]:
    config = header.get("agent_config", {})
    layers = config.get("layers")
    if isinstance(layers, dict) and isinstance(layers.get("planner"), dict):
        return dict(layers["planner"])
    return dict(config)


def _usage_summary(report: dict[str, Any], *, layered: bool) -> dict[str, Any]:
    usage = report["model_usage"]
    planner = usage.get("by_role", {}).get("planner", {})
    planner_calls = int(
        planner.get("call_count", 0) if layered else usage.get("call_count", 0)
    )
    return {
        "safe_strict_success_rate": report["safe_strict_success_rate"],
        "goal_completion_rate": report["goal_completion_rate"],
        "safe_execution_rate": report["safe_execution_rate"],
        "mean_milestone_coverage": report["mean_milestone_coverage"],
        "minefield_violation_rate": report["minefield_violation_rate"],
        "planner_call_count": planner_calls,
        "total_model_call_count": usage["call_count"],
        "input_tokens": usage["input_tokens"],
        "generation_ms": sum(
            float(item.get("generation_ms", 0.0))
            for item in usage.get("by_role", {}).values()
        ),
        "end_to_end_latency_p50_ms": report["end_to_end_latency_p50_ms"],
        "end_to_end_latency_p95_ms": report["end_to_end_latency_p95_ms"],
        "fast_path": usage.get("fast_path"),
    }


def build_composition_paired_comparison(
    *,
    candidate_path: str | Path,
    baseline_path: str | Path,
    bootstrap_samples: int = 10_000,
    seed: int = 20260904,
) -> dict[str, Any]:
    """Compare layered and direct conditions on an identical frozen task slice."""
    if bootstrap_samples < 100:
        raise ValueError("paired bootstrap requires at least 100 samples")
    candidate_header, candidate_trials = load_composition_run(candidate_path)
    baseline_header, baseline_trials = load_composition_run(baseline_path)
    if candidate_header.get("task_suite_sha256") != baseline_header.get(
        "task_suite_sha256"
    ):
        raise ValueError("paired composition comparison requires the same task suite")
    if candidate_header.get("task_ids_sha256") != baseline_header.get(
        "task_ids_sha256"
    ):
        raise ValueError("paired composition comparison requires the same task IDs")
    if candidate_header.get("split") != baseline_header.get("split"):
        raise ValueError("paired composition comparison requires the same split")
    if candidate_header.get("source", {}).get("tree_sha256") != baseline_header.get(
        "source", {}
    ).get("tree_sha256"):
        raise ValueError("paired composition comparison requires the same source tree")
    if candidate_header.get("harness_config") != baseline_header.get("harness_config"):
        raise ValueError("paired composition comparison requires the same Harness config")

    candidate_by_id = {trial["task_id"]: trial for trial in candidate_trials}
    baseline_by_id = {trial["task_id"]: trial for trial in baseline_trials}
    if (
        len(candidate_by_id) != len(candidate_trials)
        or len(baseline_by_id) != len(baseline_trials)
        or set(candidate_by_id) != set(baseline_by_id)
    ):
        raise ValueError("paired composition comparison requires unique paired tasks")

    candidate_planner = _planner_config(candidate_header)
    baseline_planner = _planner_config(baseline_header)
    planner_field_checks = {
        field: candidate_planner.get(field) == baseline_planner.get(field)
        for field in _PLANNER_FIELDS
    }
    protocol_match = candidate_header.get("agent_config", {}).get(
        "protocol"
    ) == baseline_header.get("agent_config", {}).get("protocol")
    checks = {
        "source_tree_match": True,
        "harness_config_match": True,
        "task_suite_match": True,
        "task_ids_match": True,
        "protocol_match": protocol_match,
        "planner_identity_match": all(planner_field_checks.values()),
        "planner_fields": planner_field_checks,
    }
    checks["strict_protocol_match"] = bool(
        checks["protocol_match"] and checks["planner_identity_match"]
    )
    if not checks["strict_protocol_match"]:
        raise ValueError("paired composition planner or protocol identity differs")

    task_ids = tuple(sorted(candidate_by_id))
    strict_deltas = [
        int(bool(candidate_by_id[task_id]["grade"]["success"]))
        - int(bool(baseline_by_id[task_id]["grade"]["success"]))
        for task_id in task_ids
    ]
    milestone_deltas = [
        float(candidate_by_id[task_id]["grade"]["diagnostics"]["milestone_coverage"])
        - float(baseline_by_id[task_id]["grade"]["diagnostics"]["milestone_coverage"])
        for task_id in task_ids
    ]
    rng = random.Random(seed)
    sampled = sorted(
        mean(strict_deltas[rng.randrange(len(strict_deltas))] for _ in strict_deltas)
        for _ in range(bootstrap_samples)
    )

    def percentile(fraction: float) -> float:
        return sampled[round((len(sampled) - 1) * fraction)]

    candidate_report = build_composition_report(candidate_path)
    baseline_report = build_composition_report(baseline_path)
    candidate_metrics = _usage_summary(candidate_report, layered=True)
    baseline_metrics = _usage_summary(baseline_report, layered=False)
    return {
        "format_version": "deviceops-ul-v2-composition-paired-comparison-v0.1",
        "split": candidate_header["split"],
        "task_count": len(task_ids),
        "task_suite_sha256": candidate_header["task_suite_sha256"],
        "candidate_agent_id": candidate_header["agent_id"],
        "baseline_agent_id": baseline_header["agent_id"],
        "bootstrap": {
            "method": "paired-percentile",
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "protocol_checks": checks,
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "paired_delta": {
            "safe_strict_success_rate": mean(strict_deltas),
            "ci95_low": percentile(0.025),
            "ci95_high": percentile(0.975),
            "mean_milestone_coverage": mean(milestone_deltas),
            "wins": sum(value > 0 for value in strict_deltas),
            "ties": sum(value == 0 for value in strict_deltas),
            "losses": sum(value < 0 for value in strict_deltas),
            "planner_call_count": (
                candidate_metrics["planner_call_count"]
                - baseline_metrics["planner_call_count"]
            ),
            "total_model_call_count": (
                candidate_metrics["total_model_call_count"]
                - baseline_metrics["total_model_call_count"]
            ),
            "input_tokens": (
                candidate_metrics["input_tokens"] - baseline_metrics["input_tokens"]
            ),
            "generation_ms": (
                candidate_metrics["generation_ms"] - baseline_metrics["generation_ms"]
            ),
        },
    }


def write_composition_paired_comparison(
    *,
    candidate_path: str | Path,
    baseline_path: str | Path,
    output_path: str | Path,
    bootstrap_samples: int = 10_000,
    seed: int = 20260904,
) -> dict[str, Any]:
    result = build_composition_paired_comparison(
        candidate_path=candidate_path,
        baseline_path=baseline_path,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite composition paired comparison")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result
