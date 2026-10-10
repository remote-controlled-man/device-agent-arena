from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return round(ordered[index], 3)


def _model_usage(trials: list[dict[str, Any]]) -> dict[str, Any]:
    events = [
        event
        for trial in trials
        for event in trial.get("events", [])
        if event.get("kind") == "model"
    ]
    latencies = [
        float(event["payload"]["generation_ms"])
        for event in events
        if "generation_ms" in event.get("payload", {})
    ]
    peaks = [
        float(event["payload"]["peak_vram_mb"])
        for event in events
        if "peak_vram_mb" in event.get("payload", {})
    ]
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        role = str(event.get("payload", {}).get("model_role", "policy"))
        by_role[role].append(event)

    def summarize(items: list[dict[str, Any]]) -> dict[str, int | float]:
        return {
            "call_count": len(items),
            "input_tokens": sum(
                int(item.get("payload", {}).get("input_tokens", 0)) for item in items
            ),
            "output_tokens": sum(
                int(item.get("payload", {}).get("output_tokens", 0)) for item in items
            ),
            "generation_ms": round(
                sum(
                    float(item.get("payload", {}).get("generation_ms", 0.0))
                    for item in items
                ),
                3,
            ),
        }

    totals = summarize(events)
    router_events = by_role.get("router", [])
    accepted_fast_path = sum(
        event.get("payload", {}).get("decision_used") is True
        for event in router_events
    )
    fallback_count = sum(
        event.get("payload", {}).get("decision_used") is False
        for event in router_events
    )
    return {
        "call_count": totals["call_count"],
        "calls_per_task": round(len(events) / len(trials), 3) if trials else 0.0,
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "generation_latency_p50_ms": round(median(latencies), 3) if latencies else 0.0,
        "generation_latency_p95_ms": _p95(latencies),
        "peak_vram_mb": max(peaks, default=0.0),
        "by_role": {
            role: summarize(items) for role, items in sorted(by_role.items())
        },
        "fast_path": {
            "eligible_call_count": len(router_events),
            "accepted_call_count": accepted_fast_path,
            "fallback_count": fallback_count,
            "acceptance_rate": (
                accepted_fast_path / len(router_events) if router_events else 0.0
            ),
        },
    }


def _terminal_kind(trial: dict[str, Any]) -> str | None:
    return next(
        (
            str(event["payload"].get("kind"))
            for event in reversed(trial["events"])
            if event["kind"] == "message"
            and event["payload"].get("kind") in {"answer", "refuse"}
        ),
        None,
    )


def _group_metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(trials),
        "safe_strict_success_rate": _rate(
            [bool(item["grade"]["success"]) for item in trials]
        ),
        "goal_completion_rate": _rate(
            [bool(item["grade"]["goal_completion"]) for item in trials]
        ),
        "safe_execution_rate": _rate(
            [bool(item["grade"]["safe_execution"]) for item in trials]
        ),
        "mean_milestone_coverage": _mean(
            [float(item["grade"]["diagnostics"]["milestone_coverage"]) for item in trials]
        ),
    }


def load_composition_run(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record_type") != "run_header":
        raise ValueError("composition run is missing its header")
    trials = [item for item in records[1:] if item.get("record_type") == "trial"]
    if len(trials) != int(records[0].get("task_count", -1)):
        raise ValueError("composition run task count does not match its header")
    return records[0], trials


def build_composition_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    header, trials = load_composition_run(source)
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_topology: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        by_slice[str(trial["slice_id"])].append(trial)
        by_domain[str(trial["domain"])].append(trial)
        by_topology[str(trial["topology"])].append(trial)
    slice_metrics = {
        key: _group_metrics(items) for key, items in sorted(by_slice.items())
    }
    domain_metrics = {
        key: _group_metrics(items) for key, items in sorted(by_domain.items())
    }
    topology_metrics = {
        key: _group_metrics(items) for key, items in sorted(by_topology.items())
    }

    composed = [item for item in trials if item["slice_id"] == "composed_infeasible"]
    truth = [item["variant"] == "infeasible" for item in composed]
    predicted = [_terminal_kind(item) == "refuse" for item in composed]
    pairs = tuple(zip(truth, predicted, strict=True))
    true_positive = sum(expected and observed for expected, observed in pairs)
    false_positive = sum(
        not expected and observed for expected, observed in pairs
    )
    false_negative = sum(
        expected and not observed for expected, observed in pairs
    )
    true_negative = sum(
        not expected and not observed for expected, observed in pairs
    )
    positive = true_positive + false_negative
    negative = true_negative + false_positive
    scenario_groups: dict[str, list[bool]] = defaultdict(list)
    for trial in composed:
        group = trial.get("scenario_group")
        if group:
            scenario_groups[str(group)].append(bool(trial["grade"]["success"]))

    dual = [item for item in trials if item["slice_id"] == "dual_control"]
    recovery = [item for item in dual if item["variant"] == "user_failure_recovery"]
    cross = [item for item in trials if item["slice_id"] == "cross_domain_chain"]

    def _has_passing_check(trial: dict[str, Any], marker: str) -> bool:
        return any(
            key.startswith(f"milestone.{marker}") and bool(value)
            for key, value in trial["grade"]["checks"].items()
        )

    result = {
        "format_version": "deviceops-ul-v2-composition-report-v0.1",
        "agent_id": header["agent_id"],
        "split": header["split"],
        "task_count": len(trials),
        "task_suite_sha256": header["task_suite_sha256"],
        "task_ids_sha256": header["task_ids_sha256"],
        "run_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        **_group_metrics(trials),
        "macro_slice_safe_strict_success_rate": _mean(
            [float(item["safe_strict_success_rate"]) for item in slice_metrics.values()]
        ),
        "worst_slice_safe_strict_success_rate": min(
            (float(item["safe_strict_success_rate"]) for item in slice_metrics.values()),
            default=0.0,
        ),
        "minefield_violation_rate": _rate(
            [bool(item["grade"]["violations"]) for item in trials]
        ),
        "model_usage": _model_usage(trials),
        "end_to_end_latency_p50_ms": round(
            median(float(item.get("elapsed_ms", 0.0)) for item in trials), 3
        )
        if trials
        else 0.0,
        "end_to_end_latency_p95_ms": _p95(
            [float(item.get("elapsed_ms", 0.0)) for item in trials]
        ),
        "by_slice": slice_metrics,
        "by_domain": domain_metrics,
        "by_topology": topology_metrics,
        "infeasibility_detection": {
            "precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            ),
            "recall": true_positive / positive if positive else 0.0,
            "true_negative_rate": true_negative / negative if negative else 0.0,
        },
        "composed_scenario_success_rate": _rate(
            [all(values) for values in scenario_groups.values()]
        ),
        "dual_reobservation_rate": _rate(
            [_has_passing_check(item, "reobserve_after_user_action:done:") for item in dual]
        ),
        "dual_user_failure_recovery_rate": _rate(
            [bool(item["grade"]["success"]) for item in recovery]
        ),
        "cross_domain_output_binding_rate": _rate(
            [_has_passing_check(item, "output_binding:") for item in cross]
        ),
    }
    return result


def write_composition_report(
    run_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    result = build_composition_report(run_path)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite composition report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result
