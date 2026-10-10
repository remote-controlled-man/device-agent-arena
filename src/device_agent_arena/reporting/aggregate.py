from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from device_agent_arena.artifacts.run_io import load_run


def _rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _write_utf8_lf(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return round(ordered[index], 3)


def _category_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(trials),
        "task_success_rate": _rate([trial["grade"]["success"] for trial in trials]),
    }


def _tool_call_efficiency(trials: list[dict[str, Any]]) -> dict[str, float | int]:
    total_calls = 0
    redundant_calls = 0
    calls_per_trial: list[int] = []
    for trial in trials:
        successful_signatures: set[str] = set()
        pending_signature: str | None = None
        trial_calls = 0
        for event in trial["events"]:
            if event["kind"] == "tool_call":
                pending_signature = json.dumps(
                    {
                        "name": event["payload"].get("name"),
                        "arguments": event["payload"].get("arguments", {}),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                total_calls += 1
                trial_calls += 1
                if pending_signature in successful_signatures:
                    redundant_calls += 1
            elif event["kind"] == "tool_result" and pending_signature is not None:
                if event["payload"].get("ok"):
                    successful_signatures.add(pending_signature)
                pending_signature = None
        calls_per_trial.append(trial_calls)
    return {
        "tool_call_count": total_calls,
        "mean_tool_calls": round(mean(calls_per_trial), 3) if calls_per_trial else 0.0,
        "redundant_tool_call_rate": (
            round(redundant_calls / total_calls, 6) if total_calls else 0.0
        ),
    }


def build_report(run_path: str | Path) -> dict[str, Any]:
    header, trial_tuple = load_run(run_path)
    trials = list(trial_tuple)
    by_category_trials: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        for category in trial["categories"]:
            by_category_trials[category].append(trial)

    tool_results = [
        event for trial in trials for event in trial["events"] if event["kind"] == "tool_result"
    ]
    model_events = [
        event for trial in trials for event in trial["events"] if event["kind"] == "model"
    ]
    model_generation_ms = [
        float(event["payload"]["generation_ms"])
        for event in model_events
        if "generation_ms" in event["payload"]
    ]
    peak_vram_mb = [
        float(event["payload"]["peak_vram_mb"])
        for event in model_events
        if "peak_vram_mb" in event["payload"]
    ]
    irrelevant_trials = by_category_trials.get("irrelevant", [])
    overtriggers = []
    for trial in irrelevant_trials:
        overtriggers.append(
            # Every device-tool attempt is an over-trigger on an irrelevant task,
            # including read-only state inspection and malformed/unknown calls.
            any(event["kind"] == "tool_call" for event in trial["events"])
        )

    clarification_checks = [
        value
        for trial in by_category_trials.get("missing_parameter", [])
        for key, value in trial["grade"]["checks"].items()
        if key.startswith("required_clarification.")
    ]
    latencies = [float(trial["elapsed_ms"]) for trial in trials]
    tool_efficiency = _tool_call_efficiency(trials)
    tool_selection_outcomes = []
    for trial in trials:
        selection_checks = [
            value
            for key, value in trial["grade"]["checks"].items()
            if key.startswith(("required_tool.", "forbidden_tool."))
        ]
        if selection_checks:
            tool_selection_outcomes.append(all(selection_checks))
    report = {
        "format_version": "0.2",
        "agent_id": header["agent_id"],
        "display_name": header["display_name"],
        "split": header["split"],
        "task_version": header.get("task_version"),
        "task_suite_sha256": header.get("task_suite_sha256"),
        "task_ids_sha256": header.get("task_ids_sha256"),
        "grader_version": header.get("grader_version"),
        "harness_config": header.get("harness_config", header.get("config")),
        "source": header.get("source"),
        "runtime": header.get("runtime"),
        "agent_config": header.get("agent_config"),
        "task_count": len(trials),
        "task_success_rate": _rate([trial["grade"]["success"] for trial in trials]),
        "tool_selection_accuracy": _rate(tool_selection_outcomes),
        "argument_validity_rate": _rate(
            [
                event["payload"].get("code") not in {"invalid_arguments", "unknown_tool"}
                for event in tool_results
            ]
        ),
        "overtrigger_rate": _rate(overtriggers),
        "clarification_rate": _rate(clarification_checks),
        "recovery_rate": _rate(
            [trial["grade"]["success"] for trial in by_category_trials.get("tool_failure", [])]
        ),
        "state_dependency_success_rate": _rate(
            [trial["grade"]["success"] for trial in by_category_trials.get("state_dependency", [])]
        ),
        "mean_steps": round(
            mean(max((event["step"] for event in trial["events"]), default=0) for trial in trials),
            3,
        ),
        "tool_error_rate": _rate([not bool(event["payload"].get("ok")) for event in tool_results]),
        **tool_efficiency,
        "latency_p50_ms": round(median(latencies), 3) if latencies else 0.0,
        "latency_p95_ms": _p95(latencies),
        "input_tokens": sum(int(event["payload"].get("input_tokens", 0)) for event in model_events),
        "output_tokens": sum(
            int(event["payload"].get("output_tokens", 0)) for event in model_events
        ),
        "model_generation_p50_ms": (
            round(median(model_generation_ms), 3) if model_generation_ms else 0.0
        ),
        "model_generation_p95_ms": _p95(model_generation_ms),
        "peak_vram_mb": round(max(peak_vram_mb), 3) if peak_vram_mb else 0.0,
        "by_category": {
            category: _category_summary(category_trials)
            for category, category_trials in sorted(by_category_trials.items())
        },
        "failures": [
            {
                "task_id": trial["task_id"],
                "categories": trial["categories"],
                "failed_checks": [
                    key for key, value in trial["grade"]["checks"].items() if not value
                ],
                "termination": trial["termination"],
            }
            for trial in trials
            if not trial["grade"]["success"]
        ],
    }
    return report


def report_markdown(report: dict[str, Any], *, language: str = "en") -> str:
    if language == "zh-CN":
        return _report_markdown_zh_cn(report)
    if language != "en":
        raise ValueError("language must be en or zh-CN")
    lines = [
        f"# {report['display_name']} — {report['split']}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Tasks | {report['task_count']} |",
        f"| Task success | {report['task_success_rate']:.3f} |",
        f"| Tool selection | {report['tool_selection_accuracy']:.3f} |",
        f"| Argument validity | {report['argument_validity_rate']:.3f} |",
        f"| Over-trigger | {report['overtrigger_rate']:.3f} |",
        f"| Clarification | {report['clarification_rate']:.3f} |",
        f"| Recovery | {report['recovery_rate']:.3f} |",
        f"| Mean steps | {report['mean_steps']:.3f} |",
        f"| Mean tool calls | {report['mean_tool_calls']:.3f} |",
        f"| Redundant tool-call rate | {report['redundant_tool_call_rate']:.3f} |",
        f"| Input tokens | {report['input_tokens']} |",
        f"| Output tokens | {report['output_tokens']} |",
        f"| Model generation p50 | {report['model_generation_p50_ms']:.1f} ms |",
        f"| Model generation p95 | {report['model_generation_p95_ms']:.1f} ms |",
        f"| End-to-end latency p50 | {report['latency_p50_ms']:.1f} ms |",
        f"| End-to-end latency p95 | {report['latency_p95_ms']:.1f} ms |",
        f"| Tool error rate | {report['tool_error_rate']:.3f} |",
        f"| State-dependency success | {report['state_dependency_success_rate']:.3f} |",
        f"| Peak VRAM | {report['peak_vram_mb']:.1f} MiB |",
        "",
        "## By category",
        "",
        "| Category | Tasks | Success |",
        "|---|---:|---:|",
    ]
    for category, summary in report["by_category"].items():
        lines.append(
            f"| {category} | {summary['task_count']} | {summary['task_success_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This benchmark uses a deterministic simulator and text-level ASR perturbations. "
            "It does not establish acoustic or real-hardware robustness.",
            "",
        ]
    )
    return "\n".join(lines)


def _report_markdown_zh_cn(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['display_name']} — {report['split']}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 任务数 | {report['task_count']} |",
        f"| 任务成功率 | {report['task_success_rate']:.3f} |",
        f"| 工具选择准确率 | {report['tool_selection_accuracy']:.3f} |",
        f"| 参数有效率 | {report['argument_validity_rate']:.3f} |",
        f"| 过度触发率 | {report['overtrigger_rate']:.3f} |",
        f"| 澄清成功率 | {report['clarification_rate']:.3f} |",
        f"| 故障恢复率 | {report['recovery_rate']:.3f} |",
        f"| 平均步数 | {report['mean_steps']:.3f} |",
        f"| 平均工具调用数 | {report['mean_tool_calls']:.3f} |",
        f"| 冗余工具调用率 | {report['redundant_tool_call_rate']:.3f} |",
        f"| 输入 Token 数 | {report['input_tokens']} |",
        f"| 输出 Token 数 | {report['output_tokens']} |",
        f"| 模型生成时延 p50 | {report['model_generation_p50_ms']:.1f} ms |",
        f"| 模型生成时延 p95 | {report['model_generation_p95_ms']:.1f} ms |",
        f"| 端到端时延 p50 | {report['latency_p50_ms']:.1f} ms |",
        f"| 端到端时延 p95 | {report['latency_p95_ms']:.1f} ms |",
        f"| 工具错误率 | {report['tool_error_rate']:.3f} |",
        f"| 状态依赖任务成功率 | {report['state_dependency_success_rate']:.3f} |",
        f"| 峰值显存 | {report['peak_vram_mb']:.1f} MiB |",
        "",
        "## 分类结果",
        "",
        "| 类别 | 任务数 | 成功率 |",
        "|---|---:|---:|",
    ]
    for category, summary in report["by_category"].items():
        lines.append(
            f"| {category} | {summary['task_count']} | {summary['task_success_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 局限性",
            "",
            "本基准使用确定性模拟器和文本级 ASR 扰动，不能证明模型具备声学鲁棒性或真实硬件鲁棒性。",
            "",
        ]
    )
    return "\n".join(lines)


def write_failure_slice(run_path: str | Path, output_path: str | Path) -> int:
    source = Path(run_path)
    _, trials = load_run(source)
    failures = []
    for trial in trials:
        if trial["grade"]["success"]:
            continue
        failures.append(
            {
                "task_id": trial["task_id"],
                "categories": trial["categories"],
                "termination": trial["termination"],
                "failed_checks": [
                    key for key, value in trial["grade"]["checks"].items() if not value
                ],
                "tool_error_codes": [
                    event["payload"].get("code")
                    for event in trial["events"]
                    if event["kind"] == "tool_result" and not event["payload"].get("ok")
                ],
                "trajectory_reference": f"{source.as_posix()}#{trial['task_id']}",
            }
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(failure, ensure_ascii=False, sort_keys=True) for failure in failures
    )
    _write_utf8_lf(destination, payload + ("\n" if failures else ""))
    return len(failures)


def write_report(
    *,
    run_path: str | Path,
    json_output: str | Path,
    markdown_output: str | Path,
    language: str = "en",
) -> dict[str, Any]:
    report = build_report(run_path)
    json_path = Path(json_output)
    markdown_path = Path(markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _write_utf8_lf(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_utf8_lf(markdown_path, report_markdown(report, language=language))
    return report
