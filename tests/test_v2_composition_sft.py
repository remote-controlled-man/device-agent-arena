from __future__ import annotations

import json
from pathlib import Path

from device_agent_arena.v2.composition_agents import FinishImmediatelyCompositionAgent
from device_agent_arena.v2.composition_io import load_composition_tasks
from device_agent_arena.v2.composition_runner import run_composition_tasks
from device_agent_arena.v2.composition_sft import (
    _recovery_records_for_composition_task,
    export_functiongemma_composition_recovery_sft,
    export_functiongemma_composition_sft,
    export_qwen_composition_curriculum_sft,
    export_qwen_failure_refinement_sft,
)
from device_agent_arena.v2.composition_suite import write_composition_suite


def test_composition_sft_exports_every_train_only_oracle_decision(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    output = tmp_path / "composition.jsonl"
    manifest_path = tmp_path / "composition.manifest.json"
    write_composition_suite(
        suite,
        root_seed=20260902,
        train_count=18,
        dev_count=18,
        test_count=36,
    )
    train = load_composition_tasks(suite / "train.jsonl")
    expected = sum(len(task.oracle_plan) for task in train)

    manifest = export_functiongemma_composition_sft(
        suite_root=suite,
        output=output,
        manifest_output=manifest_path,
        composition_repeats=1,
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert manifest["record_count"] == expected
    assert manifest["sha256"] == manifest["data_sha256"]
    assert len(records) == expected
    assert {record["metadata"]["split"] for record in records} == {"train"}
    assert {record["metadata"]["slice_id"] for record in records} == {
        "dual_control",
        "composed_infeasible",
        "cross_domain_chain",
    }
    assert {record["metadata"]["domain"] for record in records} == {
        "home",
        "vehicle",
        "workspace",
    }
    assert all(record["prompt"][0]["role"] == "developer" for record in records)
    assert all("hidden-v2c" not in json.dumps(record) for record in records)
    tool_names = {
        tool["function"]["name"] for tool in records[0]["tools"]
    }
    assert "inspect_target" in tool_names
    assert "request_user_action" in tool_names
    assert len(tool_names) == len(records[0]["tools"])


def _completion_name(record: dict) -> str:
    return str(record["completion"][0]["tool_calls"][0]["function"]["name"])


def _prompt_result_codes(record: dict) -> set[str]:
    return {
        str(message["content"]["response"]["code"])
        for message in record["prompt"]
        if message["role"] == "tool"
    }


def test_recovery_sft_labels_public_error_states_with_safe_replans(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    write_composition_suite(
        suite,
        root_seed=20260902,
        train_count=18,
        dev_count=18,
        test_count=36,
    )
    tasks = load_composition_tasks(suite / "train.jsonl")
    by_slice = {task.slice_id: task for task in tasks}

    dual = _recovery_records_for_composition_task(by_slice["dual_control"])
    cross = _recovery_records_for_composition_task(by_slice["cross_domain_chain"])
    infeasible = _recovery_records_for_composition_task(
        next(
            task
            for task in tasks
            if task.slice_id == "composed_infeasible"
            and task.provenance["variant"] == "infeasible"
        )
    )

    assert {record["metadata"]["recovery_kind"] for record in dual} >= {
        "after_first_precondition_failure",
        "after_repeated_precondition_failure",
    }
    assert {_completion_name(record) for record in dual} == {"request_user_action"}
    assert {_completion_name(record) for record in cross} == {"inspect_target"}
    assert {_completion_name(record) for record in infeasible} == {"refuse_task"}
    assert all("precondition_failed" in _prompt_result_codes(record) for record in dual)
    assert any(
        "user_action_unavailable" in _prompt_result_codes(record)
        for record in infeasible
    )


def test_composition_recovery_export_is_train_only_and_keeps_replay(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    output = tmp_path / "recovery.jsonl"
    manifest_path = tmp_path / "recovery.manifest.json"
    replay_path = tmp_path / "replay.jsonl"
    write_composition_suite(
        suite,
        root_seed=20260902,
        train_count=18,
        dev_count=18,
        test_count=36,
    )
    replay_path.write_text(
        json.dumps(
            {
                "prompt": [],
                "completion": [],
                "tools": [],
                "metadata": {"task_id": "v2-train-replay"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = export_functiongemma_composition_recovery_sft(
        suite_root=suite,
        output=output,
        manifest_output=manifest_path,
        hard_state_repeats=2,
        replay_data=replay_path,
        replay_modulus=1,
    )

    assert manifest["record_count"] > manifest["replay_record_count"] == 1
    assert manifest["hard_state_repeats"] == 2
    assert manifest["sha256"] == manifest["data_sha256"]
    assert manifest["dev_data_used"] is False
    assert manifest["test_data_used"] is False


def test_qwen_composition_curriculum_matches_tool_native_runtime(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    output = tmp_path / "qwen-composition.jsonl"
    manifest_path = tmp_path / "qwen-composition.manifest.json"
    write_composition_suite(
        suite,
        root_seed=20260902,
        train_count=18,
        dev_count=18,
        test_count=36,
    )

    manifest = export_qwen_composition_curriculum_sft(
        suite_root=suite,
        output=output,
        manifest_output=manifest_path,
        oracle_repeats=1,
        recovery_repeats=1,
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert manifest["record_count"] == len(records)
    assert manifest["oracle_source_record_count"] > 0
    assert manifest["recovery_source_record_count"] > 0
    assert manifest["dev_data_used"] is False
    assert manifest["test_data_used"] is False
    assert all(record["prompt"][0]["role"] == "system" for record in records)
    assert all(
        "V2 Composition 协议" in record["prompt"][0]["content"]
        for record in records
    )
    assert all(
        record["metadata"]["protocol"] == "qwen-tool-native-v2-composition"
        for record in records
    )
    assert all("hidden-v2c" not in json.dumps(record) for record in records)


def test_qwen_failure_refinement_replays_only_failed_train_states(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    base_data = tmp_path / "base.jsonl"
    base_manifest = tmp_path / "base.manifest.json"
    run_path = tmp_path / "failed-train.jsonl"
    output = tmp_path / "refinement.jsonl"
    manifest_path = tmp_path / "refinement.manifest.json"
    write_composition_suite(
        suite,
        root_seed=20260902,
        train_count=18,
        dev_count=18,
        test_count=36,
    )
    export_qwen_composition_curriculum_sft(
        suite_root=suite,
        output=base_data,
        manifest_output=base_manifest,
        oracle_repeats=1,
        recovery_repeats=1,
    )
    train = load_composition_tasks(suite / "train.jsonl")
    run_composition_tasks(
        tasks=train[:3],
        agent_factory=lambda _task: FinishImmediatelyCompositionAgent(),
        agent_id="failure-miner-test",
        split="train",
        output=run_path,
    )

    manifest = export_qwen_failure_refinement_sft(
        suite_root=suite,
        failure_run=run_path,
        replay_data=base_data,
        output=output,
        manifest_output=manifest_path,
        correction_repeats=2,
        hard_oracle_repeats=1,
        recovery_repeats=1,
        replay_modulus=8,
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    corrections = [
        record
        for record in records
        if record["metadata"]["curriculum_source"] == "train-failure-correction"
    ]
    assert manifest["record_count"] == len(records)
    assert manifest["failed_task_count"] == 3
    assert manifest["correction_source_record_count"] == 3
    assert len(corrections) == 6
    assert all(
        record["metadata"]["observed_wrong_action"] == "finish_task"
        for record in corrections
    )
    assert all(record["metadata"]["split"] == "train" for record in records)
    assert manifest["dev_data_used"] is False
    assert manifest["test_data_used"] is False
    assert manifest["prompt_conflict_count"] == 0
    assert all("hidden-v2c" not in json.dumps(record) for record in records)
