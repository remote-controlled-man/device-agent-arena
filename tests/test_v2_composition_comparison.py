from __future__ import annotations

import json

import pytest

from device_agent_arena.v2.composition_agents import (
    OracleCompositionAgent,
    RefuseImmediatelyCompositionAgent,
)
from device_agent_arena.v2.composition_comparison import (
    build_composition_paired_comparison,
)
from device_agent_arena.v2.composition_generate import generate_composition_tasks
from device_agent_arena.v2.composition_runner import run_composition_tasks


def _metadata(*, layered: bool) -> dict:
    planner = {
        "model_id": "qwen",
        "model_artifact_sha256": "qwen-tree",
        "adapter_path": "qwen-adapter",
        "adapter_sha256": "qwen-adapter-hash",
        "temperature": 0.0,
        "max_new_tokens": 256,
        "load_in_4bit": True,
        "chat_template_sha256": "qwen-template",
    }
    if not layered:
        return {**planner, "protocol": "deviceops-ul-v2-composition-tool-native-v1"}
    return {
        "protocol": "deviceops-ul-v2-composition-tool-native-v1",
        "layers": {
            "planner": planner,
            "router": {
                "model_id": "functiongemma",
                "model_artifact_sha256": "router-tree",
            },
        },
    }


def test_composition_paired_comparison_binds_planner_and_bootstraps(tmp_path) -> None:
    tasks = generate_composition_tasks("train", count=9, root_seed=20260902)[:3]
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    run_composition_tasks(
        tasks=tasks,
        agent_factory=lambda _task: RefuseImmediatelyCompositionAgent(),
        agent_id="direct-test",
        split="train",
        output=baseline,
        run_metadata=_metadata(layered=False),
    )
    run_composition_tasks(
        tasks=tasks,
        agent_factory=OracleCompositionAgent,
        agent_id="layered-test",
        split="train",
        output=candidate,
        run_metadata=_metadata(layered=True),
    )

    result = build_composition_paired_comparison(
        candidate_path=candidate,
        baseline_path=baseline,
        bootstrap_samples=1000,
        seed=7,
    )

    assert result["protocol_checks"]["strict_protocol_match"] is True
    assert result["protocol_checks"]["planner_identity_match"] is True
    assert result["candidate"]["safe_strict_success_rate"] == 1.0
    assert result["baseline"]["safe_strict_success_rate"] == 0.0
    assert result["paired_delta"]["safe_strict_success_rate"] == 1.0
    assert result["bootstrap"] == {
        "method": "paired-percentile",
        "samples": 1000,
        "seed": 7,
    }


def test_composition_paired_comparison_rejects_different_source_tree(tmp_path) -> None:
    tasks = generate_composition_tasks("train", count=9, root_seed=20260902)[:1]
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    for path in (baseline, candidate):
        run_composition_tasks(
            tasks=tasks,
            agent_factory=OracleCompositionAgent,
            agent_id=path.stem,
            split="train",
            output=path,
            run_metadata=_metadata(layered=path is candidate),
        )
    records = [
        json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["source"]["tree_sha256"] = "changed"
    candidate.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="source tree"):
        build_composition_paired_comparison(
            candidate_path=candidate,
            baseline_path=baseline,
        )
