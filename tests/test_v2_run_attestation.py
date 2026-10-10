from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from device_agent_arena.v2.run_attestation import (
    verify_layered_model_run_attestation,
    verify_model_run_attestation,
)


def _write_qwen_run(path: Path, *, executed_target: str = "node-1") -> None:
    raw = (
        '<tool_call>\n{"name":"inspect_target","arguments":'
        '{"target_id":"node-1"}}\n</tool_call><|im_end|>'
    )
    task_id = "task-1"
    records = [
        {
            "record_type": "run_header",
            "task_ids_sha256": hashlib.sha256(f"{task_id}\n".encode()).hexdigest(),
            "agent_config": {
                "attestation_version": "deviceops-ul-model-run-attestation-v2",
                "execution_authority": "deviceops-ul-transformers-runner-v1",
                "protocol": "deviceops-ul-v2-composition-tool-native-v1",
                "model_id": "qwen-local",
                "model_artifact_sha256": "checkpoint-hash",
                "temperature": 0.0,
                "max_new_tokens": 256,
                "load_in_4bit": True,
                "chat_template_sha256": "template-hash",
            },
        },
        {
            "record_type": "trial",
            "task_id": task_id,
            "events": [
                {
                    "step": 1,
                    "kind": "model",
                    "payload": {"raw_model_output": raw},
                },
                {
                    "step": 1,
                    "kind": "tool_call",
                    "payload": {
                        "name": "inspect_target",
                        "arguments": {"target_id": executed_target},
                    },
                },
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _verify(path: Path) -> None:
    verify_model_run_attestation(
        path,
        expected_protocol="deviceops-ul-v2-composition-tool-native-v1",
        checkpoint={"path": "qwen-local", "sha256": "checkpoint-hash"},
        expected_runtime={
            "temperature": 0.0,
            "max_new_tokens": 256,
            "load_in_4bit": True,
            "chat_template_sha256": "template-hash",
        },
    )


def test_qwen_attestation_replays_native_tool_call(tmp_path: Path) -> None:
    path = tmp_path / "qwen.jsonl"
    _write_qwen_run(path)

    _verify(path)


def test_qwen_attestation_rejects_changed_execution(tmp_path: Path) -> None:
    path = tmp_path / "qwen-forged.jsonl"
    _write_qwen_run(path, executed_target="node-2")

    with pytest.raises(ValueError, match="does not match"):
        _verify(path)


def test_qwen_attestation_binds_adapter_weights(tmp_path: Path) -> None:
    path = tmp_path / "qwen-adapter.jsonl"
    _write_qwen_run(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["agent_config"].update(
        {"adapter_path": "adapter", "adapter_sha256": "recorded-adapter-hash"}
    )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="adapter_sha256"):
        verify_model_run_attestation(
            path,
            expected_protocol="deviceops-ul-v2-composition-tool-native-v1",
            checkpoint={
                "path": "qwen-local",
                "sha256": "checkpoint-hash",
                "adapter_path": "adapter",
                "adapter_sha256": "expected-adapter-hash",
            },
            expected_runtime={
                "temperature": 0.0,
                "max_new_tokens": 256,
                "load_in_4bit": True,
                "chat_template_sha256": "template-hash",
            },
        )


def test_layered_attestation_binds_rejected_router_and_used_planner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "layered.jsonl"
    task_id = "task-1"
    router_raw = (
        '<tool_call>\n{"name":"poll_action","arguments":'
        '{"operation_id":"fake"}}\n</tool_call><|im_end|>'
    )
    planner_raw = (
        '<tool_call>\n{"name":"inspect_target","arguments":'
        '{"target_id":"node-1"}}\n</tool_call><|im_end|>'
    )
    runtime = {
        "temperature": 0.0,
        "max_new_tokens": 256,
        "load_in_4bit": True,
        "chat_template_sha256": "template-hash",
    }
    layers = {
        role: {"path": role, "sha256": f"{role}-hash", "runtime": runtime}
        for role in ("router", "planner")
    }
    records = [
        {
            "record_type": "run_header",
            "task_ids_sha256": hashlib.sha256(f"{task_id}\n".encode()).hexdigest(),
            "agent_config": {
                "attestation_version": "deviceops-ul-layered-model-run-attestation-v1",
                "execution_authority": "deviceops-ul-transformers-runner-v1",
                "protocol": "deviceops-ul-v2-composition-tool-native-v1",
                "fast_path_contract": "deviceops-ul-v2-speculative-fast-path-v1",
                "layers": {
                    role: {
                        "model_id": spec["path"],
                        "model_artifact_sha256": spec["sha256"],
                        **runtime,
                    }
                    for role, spec in layers.items()
                },
            },
        },
        {
            "record_type": "trial",
            "task_id": task_id,
            "events": [
                {
                    "step": 1,
                    "kind": "model",
                    "payload": {
                        "raw_model_output": router_raw,
                        "model_role": "router",
                        "decision_used": False,
                        "rejection_reason": "fast_path_contract_mismatch",
                    },
                },
                {
                    "step": 1,
                    "kind": "model",
                    "payload": {
                        "raw_model_output": planner_raw,
                        "model_role": "planner",
                        "decision_used": True,
                    },
                },
                {
                    "step": 1,
                    "kind": "tool_call",
                    "payload": {
                        "name": "inspect_target",
                        "arguments": {"target_id": "node-1"},
                    },
                },
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    verify_layered_model_run_attestation(
        path,
        expected_protocol="deviceops-ul-v2-composition-tool-native-v1",
        layers=layers,
    )

    records[1]["events"][1]["payload"]["decision_used"] = False
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly one used model decision"):
        verify_layered_model_run_attestation(
            path,
            expected_protocol="deviceops-ul-v2-composition-tool-native-v1",
            layers=layers,
        )
