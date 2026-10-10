from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from device_agent_arena.agents.hf_backend import (
    parse_functiongemma_response,
    parse_qwen_response,
    uses_functiongemma_protocol,
)


def verify_model_run_attestation(
    run_path: str | Path,
    *,
    expected_protocol: str,
    checkpoint: dict[str, Any],
    expected_runtime: dict[str, Any],
) -> None:
    """Verify model provenance and bind each recorded decision to raw model output."""
    records = [
        json.loads(line)
        for line in Path(run_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record_type") != "run_header":
        raise ValueError("attested run is missing its header")
    header = records[0]
    trials = [item for item in records[1:] if item.get("record_type") == "trial"]
    config = header.get("agent_config", {})
    required = {
        "attestation_version": "deviceops-ul-model-run-attestation-v2",
        "execution_authority": "deviceops-ul-transformers-runner-v1",
        "protocol": expected_protocol,
        "model_id": checkpoint["path"],
        "model_artifact_sha256": checkpoint["sha256"],
        "temperature": 0.0,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"raw run attestation mismatch: {key}")
    adapter_keys = {"adapter_path", "adapter_sha256"}
    checkpoint_has_adapter = any(checkpoint.get(key) is not None for key in adapter_keys)
    run_has_adapter = any(config.get(key) is not None for key in adapter_keys)
    if checkpoint_has_adapter != run_has_adapter:
        raise ValueError("raw run attestation has an unbound adapter")
    if checkpoint_has_adapter:
        for key in sorted(adapter_keys):
            if not checkpoint.get(key) or config.get(key) != checkpoint[key]:
                raise ValueError(f"raw run attestation mismatch: {key}")
    required_runtime_keys = {
        "temperature",
        "max_new_tokens",
        "load_in_4bit",
        "chat_template_sha256",
    }
    if set(expected_runtime) != required_runtime_keys:
        raise ValueError("gate config does not fully freeze decoder runtime")
    for key, expected in expected_runtime.items():
        if config.get(key) != expected:
            raise ValueError(f"raw run decoder mismatch: {key}")

    task_ids = [str(item.get("task_id", "")) for item in trials]
    if not all(task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("raw run contains missing or duplicate task IDs")
    observed_ids = hashlib.sha256(("\n".join(task_ids) + "\n").encode()).hexdigest()
    if observed_ids != header.get("task_ids_sha256"):
        raise ValueError("raw run task IDs do not match the header")

    parse_response = (
        parse_functiongemma_response
        if uses_functiongemma_protocol(checkpoint["path"])
        else parse_qwen_response
    )

    for trial in trials:
        by_step: dict[int, list[dict[str, Any]]] = {}
        for event in trial.get("events", []):
            by_step.setdefault(int(event["step"]), []).append(event)
        for step, events in by_step.items():
            calls = [event for event in events if event.get("kind") == "tool_call"]
            agent_messages = [
                event
                for event in events
                if event.get("kind") == "message"
                and event.get("payload", {}).get("kind")
                in {"clarify", "confirm", "answer", "refuse"}
            ]
            if not calls and not agent_messages:
                continue
            model_events = [event for event in events if event.get("kind") == "model"]
            if len(model_events) != 1:
                raise ValueError(
                    f"raw run decision at {trial['task_id']} step {step} lacks one model event"
                )
            raw = model_events[0].get("payload", {}).get("raw_model_output")
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("raw run model event is missing raw_model_output")
            if calls:
                parsed = parse_response(raw).tool_calls
                if len(parsed) != 1:
                    raise ValueError("raw model output does not contain exactly one tool call")
                recorded = calls[0]["payload"]
                if parsed[0].name != recorded.get("name") or parsed[0].arguments != recorded.get(
                    "arguments"
                ):
                    raise ValueError("raw model output does not match the executed tool call")


def _verify_layer_provenance(
    *,
    role: str,
    recorded: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    required = {
        "model_id": expected["path"],
        "model_artifact_sha256": expected["sha256"],
        **expected["runtime"],
    }
    for key, value in required.items():
        if recorded.get(key) != value:
            raise ValueError(f"layered run {role} attestation mismatch: {key}")
    adapter_keys = ("adapter_path", "adapter_sha256")
    for key in adapter_keys:
        if recorded.get(key) != expected.get(key):
            raise ValueError(f"layered run {role} attestation mismatch: {key}")


def verify_layered_model_run_attestation(
    run_path: str | Path,
    *,
    expected_protocol: str,
    layers: dict[str, dict[str, Any]],
) -> None:
    """Bind both model layers and the actually executed decision to raw outputs."""
    if set(layers) != {"router", "planner"}:
        raise ValueError("layered attestation requires router and planner bindings")
    records = [
        json.loads(line)
        for line in Path(run_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record_type") != "run_header":
        raise ValueError("attested layered run is missing its header")
    header = records[0]
    config = header.get("agent_config", {})
    required_header = {
        "attestation_version": "deviceops-ul-layered-model-run-attestation-v1",
        "execution_authority": "deviceops-ul-transformers-runner-v1",
        "protocol": expected_protocol,
        "fast_path_contract": "deviceops-ul-v2-speculative-fast-path-v1",
    }
    for key, value in required_header.items():
        if config.get(key) != value:
            raise ValueError(f"layered run attestation mismatch: {key}")
    recorded_layers = config.get("layers", {})
    if set(recorded_layers) != set(layers):
        raise ValueError("layered run model roles do not match frozen bindings")
    for role, expected in layers.items():
        _verify_layer_provenance(
            role=role,
            recorded=recorded_layers[role],
            expected=expected,
        )

    trials = [item for item in records[1:] if item.get("record_type") == "trial"]
    task_ids = [str(item.get("task_id", "")) for item in trials]
    if not all(task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("layered run contains missing or duplicate task IDs")
    observed_ids = hashlib.sha256(("\n".join(task_ids) + "\n").encode()).hexdigest()
    if observed_ids != header.get("task_ids_sha256"):
        raise ValueError("layered run task IDs do not match the header")

    for trial in trials:
        by_step: dict[int, list[dict[str, Any]]] = {}
        for event in trial.get("events", []):
            by_step.setdefault(int(event["step"]), []).append(event)
        for step, events in by_step.items():
            calls = [event for event in events if event.get("kind") == "tool_call"]
            if not calls:
                continue
            if len(calls) != 1:
                raise ValueError("layered run step must execute exactly one tool call")
            model_events = [event for event in events if event.get("kind") == "model"]
            used = [
                event
                for event in model_events
                if event.get("payload", {}).get("decision_used") is True
            ]
            if len(used) != 1:
                raise ValueError(
                    "layered run decision must have exactly one used model decision"
                )
            for event in model_events:
                payload = event.get("payload", {})
                role = str(payload.get("model_role", ""))
                if role not in layers:
                    raise ValueError("layered run model event has an unknown role")
                raw = payload.get("raw_model_output")
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError("layered run model event is missing raw output")
                if payload.get("decision_used") is not True:
                    if role != "router" or payload.get("rejection_reason") != (
                        "fast_path_contract_mismatch"
                    ):
                        raise ValueError("layered run rejected decision is not authenticated")
                    continue
                parser = (
                    parse_functiongemma_response
                    if uses_functiongemma_protocol(layers[role]["path"])
                    else parse_qwen_response
                )
                parsed = parser(raw).tool_calls
                recorded = calls[0]["payload"]
                if (
                    len(parsed) != 1
                    or parsed[0].name != recorded.get("name")
                    or parsed[0].arguments != recorded.get("arguments")
                ):
                    raise ValueError(
                        f"layered raw output does not match executed call at "
                        f"{trial['task_id']} step {step}"
                    )
