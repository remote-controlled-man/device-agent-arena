from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PROTOCOL_ABLATION_KEYS = ("visible_contract_guard",)


def _fractional_change(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return round(after / before - 1, 6)


def _write_utf8_lf(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def compare_reports(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    compatibility_fields = (
        "split",
        "task_count",
        "task_version",
        "task_suite_sha256",
        "task_ids_sha256",
        "grader_version",
        "harness_config",
    )
    for field in compatibility_fields:
        if base.get(field) is None or candidate.get(field) is None:
            raise ValueError(f"reports are missing required compatibility field: {field}")
        if base[field] != candidate[field]:
            raise ValueError(f"reports are incompatible: {field} differs")

    base_source = base.get("source") or {}
    candidate_source = candidate.get("source") or {}
    base_commit = base_source.get("commit")
    candidate_commit = candidate_source.get("commit")
    base_tree = base_source.get("tree_sha256")
    candidate_tree = candidate_source.get("tree_sha256")
    same_commit = bool(base_commit and candidate_commit and base_commit == candidate_commit)
    same_tree = bool(base_tree and candidate_tree and base_tree == candidate_tree)
    has_tree_evidence = bool(base_tree or candidate_tree)
    same_clean_commit = bool(
        same_commit
        and base_source.get("dirty") is False
        and candidate_source.get("dirty") is False
    )
    if (has_tree_evidence and not same_tree) or (
        not has_tree_evidence and not same_clean_commit
    ):
        raise ValueError(
            "reports must use the same source-tree hash, or the same clean source commit"
        )

    raw_base_protocol = (base.get("agent_config") or {}).get("comparison_protocol")
    raw_candidate_protocol = (candidate.get("agent_config") or {}).get("comparison_protocol")
    if not raw_base_protocol or not raw_candidate_protocol:
        raise ValueError("reports must use the same model/decoding/hardware protocol")
    base_protocol = {
        key: value
        for key, value in raw_base_protocol.items()
        if key not in _PROTOCOL_ABLATION_KEYS
    }
    candidate_protocol = {
        key: value
        for key, value in raw_candidate_protocol.items()
        if key not in _PROTOCOL_ABLATION_KEYS
    }
    if base_protocol != candidate_protocol:
        raise ValueError("reports must use the same model/decoding/hardware protocol")
    protocol_ablation = {
        key: {
            "base": raw_base_protocol.get(key),
            "candidate": raw_candidate_protocol.get(key),
        }
        for key in _PROTOCOL_ABLATION_KEYS
        if raw_base_protocol.get(key) != raw_candidate_protocol.get(key)
    }

    base_success = float(base["task_success_rate"])
    candidate_success = float(candidate["task_success_rate"])
    categories = sorted(set(base["by_category"]) | set(candidate["by_category"]))
    by_category = {}
    for category in categories:
        base_slice = base["by_category"].get(category, {})
        candidate_slice = candidate["by_category"].get(category, {})
        before = float(base_slice.get("task_success_rate", 0.0))
        after = float(candidate_slice.get("task_success_rate", 0.0))
        by_category[category] = {
            "task_count": candidate_slice.get("task_count", base_slice.get("task_count", 0)),
            "base": before,
            "candidate": after,
            "delta_pp": round((after - before) * 100, 3),
        }

    return {
        "format_version": "0.2",
        "split": base["split"],
        "task_count": base["task_count"],
        "task_version": base["task_version"],
        "task_suite_sha256": base["task_suite_sha256"],
        "task_ids_sha256": base["task_ids_sha256"],
        "grader_version": base["grader_version"],
        "harness_config": base["harness_config"],
        "source_commit": base_commit,
        "source_tree_sha256": base_tree,
        "comparison_protocol": base_protocol,
        **({"protocol_ablation": protocol_ablation} if protocol_ablation else {}),
        "base": base["display_name"],
        "candidate": candidate["display_name"],
        "base_task_success_rate": base_success,
        "candidate_task_success_rate": candidate_success,
        "task_success_delta_pp": round((candidate_success - base_success) * 100, 3),
        "task_success_relative_improvement": _fractional_change(base_success, candidate_success),
        "base_latency_p50_ms": base["latency_p50_ms"],
        "candidate_latency_p50_ms": candidate["latency_p50_ms"],
        "latency_p50_change": _fractional_change(
            float(base["latency_p50_ms"]), float(candidate["latency_p50_ms"])
        ),
        "latency_p95_change": _fractional_change(
            float(base["latency_p95_ms"]), float(candidate["latency_p95_ms"])
        ),
        "base_latency_p95_ms": base["latency_p95_ms"],
        "candidate_latency_p95_ms": candidate["latency_p95_ms"],
        "base_model_generation_p50_ms": base["model_generation_p50_ms"],
        "candidate_model_generation_p50_ms": candidate["model_generation_p50_ms"],
        "model_generation_p50_change": _fractional_change(
            float(base["model_generation_p50_ms"]),
            float(candidate["model_generation_p50_ms"]),
        ),
        "input_tokens_change": _fractional_change(
            float(base["input_tokens"]), float(candidate["input_tokens"])
        ),
        "output_tokens_change": _fractional_change(
            float(base["output_tokens"]), float(candidate["output_tokens"])
        ),
        "base_input_tokens": base["input_tokens"],
        "candidate_input_tokens": candidate["input_tokens"],
        "base_output_tokens": base["output_tokens"],
        "candidate_output_tokens": candidate["output_tokens"],
        "mean_steps_change": _fractional_change(
            float(base["mean_steps"]), float(candidate["mean_steps"])
        ),
        "mean_tool_calls_change": _fractional_change(
            float(base["mean_tool_calls"]), float(candidate["mean_tool_calls"])
        ),
        "base_mean_steps": base["mean_steps"],
        "candidate_mean_steps": candidate["mean_steps"],
        "base_mean_tool_calls": base["mean_tool_calls"],
        "candidate_mean_tool_calls": candidate["mean_tool_calls"],
        "base_redundant_tool_call_rate": base["redundant_tool_call_rate"],
        "candidate_redundant_tool_call_rate": candidate["redundant_tool_call_rate"],
        "base_peak_vram_mb": base["peak_vram_mb"],
        "candidate_peak_vram_mb": candidate["peak_vram_mb"],
        "by_category": by_category,
    }


def comparison_markdown(comparison: dict[str, Any], *, language: str = "en") -> str:
    if language == "zh-CN":
        return _comparison_markdown_zh_cn(comparison)
    if language != "en":
        raise ValueError("language must be en or zh-CN")
    relative = comparison["task_success_relative_improvement"]
    relative_text = "n/a" if relative is None else f"{relative * 100:.1f}%"

    def change_text(key: str) -> str:
        value = comparison[key]
        return "n/a" if value is None else f"{value * 100:+.1f}%"

    lines = [
        "# Model comparison",
        "",
        f"Split: `{comparison['split']}` · Tasks: {comparison['task_count']}",
        "",
        "| Metric | Base | Candidate | Change |",
        "|---|---:|---:|---:|",
        f"| Task success | {comparison['base_task_success_rate']:.3f} | "
        f"{comparison['candidate_task_success_rate']:.3f} | "
        f"{comparison['task_success_delta_pp']:.2f} pp ({relative_text}) |",
        f"| End-to-end latency p50 | {comparison['base_latency_p50_ms']:.1f} ms | "
        f"{comparison['candidate_latency_p50_ms']:.1f} ms | "
        f"{change_text('latency_p50_change')} |",
        f"| End-to-end latency p95 | {comparison['base_latency_p95_ms']:.1f} ms | "
        f"{comparison['candidate_latency_p95_ms']:.1f} ms | "
        f"{change_text('latency_p95_change')} |",
        f"| Model generation p50 | {comparison['base_model_generation_p50_ms']:.1f} ms | "
        f"{comparison['candidate_model_generation_p50_ms']:.1f} ms | "
        f"{change_text('model_generation_p50_change')} |",
        f"| Output tokens | {comparison['base_output_tokens']} | "
        f"{comparison['candidate_output_tokens']} | {change_text('output_tokens_change')} |",
        f"| Mean steps | {comparison['base_mean_steps']:.3f} | "
        f"{comparison['candidate_mean_steps']:.3f} | {change_text('mean_steps_change')} |",
        f"| Mean tool calls | {comparison['base_mean_tool_calls']:.3f} | "
        f"{comparison['candidate_mean_tool_calls']:.3f} | "
        f"{change_text('mean_tool_calls_change')} |",
        f"| Redundant tool-call rate | {comparison['base_redundant_tool_call_rate']:.3f} | "
        f"{comparison['candidate_redundant_tool_call_rate']:.3f} | — |",
        f"| Peak inference VRAM | {comparison['base_peak_vram_mb']:.1f} MiB | "
        f"{comparison['candidate_peak_vram_mb']:.1f} MiB | — |",
        "",
        "## By category",
        "",
        "| Category | Base | Candidate | Change |",
        "|---|---:|---:|---:|",
    ]
    for category, summary in comparison["by_category"].items():
        lines.append(
            f"| {category} | {summary['base']:.3f} | {summary['candidate']:.3f} | "
            f"{summary['delta_pp']:.2f} pp |"
        )
    lines.extend(
        [
            "",
            "The comparison is valid only when both reports use the same frozen task split, "
            "grader, decoding settings, and hardware class.",
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_markdown_zh_cn(comparison: dict[str, Any]) -> str:
    relative = comparison["task_success_relative_improvement"]
    relative_text = "不适用" if relative is None else f"{relative * 100:.1f}%"

    def change_text(key: str) -> str:
        value = comparison[key]
        return "不适用" if value is None else f"{value * 100:+.1f}%"

    lines = [
        "# 模型对比",
        "",
        f"数据切分：`{comparison['split']}` · 任务数：{comparison['task_count']}",
        "",
        "| 指标 | 基线 | 候选 | 变化 |",
        "|---|---:|---:|---:|",
        f"| 任务成功率 | {comparison['base_task_success_rate']:.3f} | "
        f"{comparison['candidate_task_success_rate']:.3f} | "
        f"{comparison['task_success_delta_pp']:.2f} 个百分点（{relative_text}） |",
        f"| 端到端时延 p50 | {comparison['base_latency_p50_ms']:.1f} ms | "
        f"{comparison['candidate_latency_p50_ms']:.1f} ms | "
        f"{change_text('latency_p50_change')} |",
        f"| 端到端时延 p95 | {comparison['base_latency_p95_ms']:.1f} ms | "
        f"{comparison['candidate_latency_p95_ms']:.1f} ms | "
        f"{change_text('latency_p95_change')} |",
        f"| 模型生成时延 p50 | {comparison['base_model_generation_p50_ms']:.1f} ms | "
        f"{comparison['candidate_model_generation_p50_ms']:.1f} ms | "
        f"{change_text('model_generation_p50_change')} |",
        f"| 输出 Token 数 | {comparison['base_output_tokens']} | "
        f"{comparison['candidate_output_tokens']} | {change_text('output_tokens_change')} |",
        f"| 平均步数 | {comparison['base_mean_steps']:.3f} | "
        f"{comparison['candidate_mean_steps']:.3f} | {change_text('mean_steps_change')} |",
        f"| 平均工具调用数 | {comparison['base_mean_tool_calls']:.3f} | "
        f"{comparison['candidate_mean_tool_calls']:.3f} | "
        f"{change_text('mean_tool_calls_change')} |",
        f"| 冗余工具调用率 | {comparison['base_redundant_tool_call_rate']:.3f} | "
        f"{comparison['candidate_redundant_tool_call_rate']:.3f} | — |",
        f"| 推理峰值显存 | {comparison['base_peak_vram_mb']:.1f} MiB | "
        f"{comparison['candidate_peak_vram_mb']:.1f} MiB | — |",
        "",
        "## 分类结果",
        "",
        "| 类别 | 基线 | 候选 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for category, summary in comparison["by_category"].items():
        lines.append(
            f"| {category} | {summary['base']:.3f} | {summary['candidate']:.3f} | "
            f"{summary['delta_pp']:.2f} 个百分点 |"
        )
    lines.extend(
        [
            "",
            "只有当两份报告使用相同的冻结数据切分、评分器、解码设置和硬件类别时，本对比才有效。",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison(
    *,
    base_path: str | Path,
    candidate_path: str | Path,
    json_output: str | Path,
    markdown_output: str | Path,
    language: str = "en",
) -> dict[str, Any]:
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    comparison = compare_reports(base, candidate)
    json_destination = Path(json_output)
    markdown_destination = Path(markdown_output)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    _write_utf8_lf(
        json_destination,
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_utf8_lf(
        markdown_destination,
        comparison_markdown(comparison, language=language),
    )
    return comparison
