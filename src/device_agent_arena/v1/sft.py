from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from device_agent_arena.domain.models import ToolCall
from device_agent_arena.v1.agents import OracleAgentV1
from device_agent_arena.v1.environment import PartialContractEnvironment
from device_agent_arena.v1.models import (
    DialogueAction,
    PlanStep,
    PublicTool,
    V1Observation,
    V1TaskSpec,
)


def _tool_schema(tool: PublicTool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def v1_system_prompt(policy_text: tuple[str, ...]) -> str:
    policy = "\n".join(f"- {item}" for item in policy_text)
    return (
        "你是一个部分可观测的 DeviceOps Agent。只能依据用户消息、公开工具定义和已返回的"
        "观察作出决定，不能猜测隐藏状态。每轮选择一种行为：调用一个工具；使用 "
        "[CLARIFY field=...] 提问；使用 [CONFIRM operation=...] 解释影响并确认；或使用 "
        "[ANSWER]/[REFUSE] 结束。工具超时后先验证，避免重复副作用。\n\n安全规则：\n"
        f"{policy}"
    )


def render_dialogue_action(action: DialogueAction) -> str:
    if action.kind == "clarify":
        return f"[CLARIFY field={action.field}] {action.content}"
    if action.kind == "confirm":
        return f"[CONFIRM operation={action.operation}] {action.content}"
    if action.kind == "refuse":
        return f"[REFUSE] {action.content}"
    return f"[ANSWER] {action.content}"


def messages_for_observation(observation: V1Observation) -> list[dict]:
    """Build the canonical public history shared by training and inference."""
    messages: list[dict] = [
        {"role": "system", "content": v1_system_prompt(observation.safety_policy)},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instruction": observation.instruction,
                    "public_context": observation.public_context,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]
    call_index = 0
    pending_calls: list[tuple[str, str]] = []
    for event in observation.events:
        if event.kind == "tool_call":
            call_index += 1
            call_id = f"call_{call_index}"
            name = str(event.payload["name"])
            pending_calls.append((call_id, name))
            messages.append(
                _assistant_message(
                    ToolCall(name, deepcopy(event.payload.get("arguments", {}))),
                    call_id=call_id,
                )
            )
        elif event.kind == "tool_result":
            if not pending_calls:
                continue
            call_id, name = pending_calls.pop(0)
            public_result = {
                key: deepcopy(event.payload.get(key))
                for key in ("ok", "code", "message", "state_delta", "data", "details")
            }
            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        public_result, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
        elif event.kind == "message" and event.payload.get("kind") in {
            "clarify",
            "confirm",
            "answer",
            "refuse",
        }:
            messages.append(
                {
                    "role": "assistant",
                    "content": render_dialogue_action(
                        DialogueAction(
                            kind=event.payload["kind"],
                            content=str(event.payload.get("content", "")),
                            field=event.payload.get("field"),
                            operation=event.payload.get("operation"),
                        )
                    ),
                }
            )
        elif event.kind == "user":
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        event.payload, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    return messages


def _assistant_message(action: ToolCall | DialogueAction, *, call_id: str) -> dict:
    if isinstance(action, ToolCall):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "arguments": json.dumps(
                            action.arguments, ensure_ascii=False, sort_keys=True
                        ),
                    },
                }
            ],
        }
    return {"role": "assistant", "content": render_dialogue_action(action)}


def _records_for_task(
    task: V1TaskSpec,
    *,
    agent_task: V1TaskSpec | None = None,
    normalize_idempotency_keys: bool = False,
    format_version: str = "deviceops-ul-sft-v1.1-next-decision",
) -> tuple[dict, ...]:
    """Export one prompt-completion sample for every oracle decision state."""
    environment = PartialContractEnvironment()
    observation = environment.reset(task)
    policy_task = agent_task or task
    agent = OracleAgentV1(policy_task)
    tools = [_tool_schema(tool) for tool in observation.tools]
    records: list[dict] = []
    call_index = 0
    decision_index = 0
    mutation_index = 0
    while True:
        action = agent.act(observation)
        decision_index += 1
        if (
            normalize_idempotency_keys
            and isinstance(action, ToolCall)
            and "idempotency_key" in action.arguments
        ):
            mutation_index += 1
            arguments = deepcopy(action.arguments)
            arguments["idempotency_key"] = (
                f"{arguments.get('action', 'change')}-"
                f"{arguments.get('target_id', 'target')}-{mutation_index}"
            )
            action = ToolCall(action.name, arguments)
        next_call_id = f"call_{call_index + 1}"
        completion = _assistant_message(action, call_id=next_call_id)
        records.append(
            {
                "prompt": messages_for_observation(observation),
                "completion": [deepcopy(completion)],
                "tools": deepcopy(tools),
                "metadata": {
                    "task_id": task.id,
                    "categories": list(task.categories),
                    "tool_profile": task.tool_profile,
                    "decision_index": decision_index,
                    "decision_kind": "tool" if isinstance(action, ToolCall) else action.kind,
                    "format_version": format_version,
                    **(
                        {"archetype": task.provenance["archetype"]}
                        if agent_task is not None
                        else {}
                    ),
                },
            }
        )
        if isinstance(action, ToolCall):
            call_index += 1
            transition = environment.step(action)
        else:
            transition = environment.step(action)
        if transition.done:
            trial = environment.finish(agent.name)
            if not trial.grade.success:
                raise RuntimeError(f"oracle SFT trajectory failed grading: {task.id}")
            break
        observation = transition.observation
    if decision_index != len(policy_task.oracle_plan):
        raise RuntimeError(
            f"oracle decision count differs from frozen plan: {task.id} "
            f"({decision_index} != {len(policy_task.oracle_plan)})"
        )
    return tuple(records)


def _grounded_policy_task(task: V1TaskSpec) -> V1TaskSpec:
    archetype = task.provenance["archetype"]
    if archetype not in {"shared_impact_accept", "shared_impact_decline"}:
        return task
    living = next(
        device["id"]
        for device in task.initial_world["devices"].values()
        if bool(device.get("shared")) and device.get("owner") == "user"
    )
    routine = next(
        item["id"]
        for item in task.initial_world["routines"].values()
        if item["device_id"] == living
    )
    confirmation = f"pause_routine:{routine}"
    plan = [
        PlanStep(kind="tool", name="search", arguments={"query": "客厅"}),
        PlanStep(kind="tool", name="inspect_device", arguments={"device_id": living}),
        PlanStep(kind="tool", name="inspect_routines", arguments={"device_id": living}),
        PlanStep(
            kind="confirm",
            operation=confirmation,
            content="这会暂停小林的共享自动化，是否继续？",
        ),
    ]
    if archetype == "shared_impact_accept":
        plan.extend(
            [
                PlanStep(
                    kind="tool",
                    name="apply_change",
                    arguments={
                        "action": "pause_routine",
                        "target_id": routine,
                        "payload": {},
                        "idempotency_key": "normalized-later",
                    },
                ),
                PlanStep(kind="tool", name="poll", arguments={"operation_id": "$last_operation"}),
                PlanStep(
                    kind="tool",
                    name="apply_change",
                    arguments={
                        "action": "set_mode",
                        "target_id": living,
                        "payload": {"mode": "cinema"},
                        "idempotency_key": "normalized-later",
                    },
                ),
                PlanStep(kind="tool", name="poll", arguments={"operation_id": "$last_operation"}),
                PlanStep(kind="answer", content="已经按确认后的方案执行完成。"),
            ]
        )
    else:
        plan.append(PlanStep(kind="refuse", content="你没有确认共享影响，因此没有执行变更。"))
    return replace(task, oracle_plan=tuple(plan))


def export_grounded_curriculum_sft(
    tasks: tuple[V1TaskSpec, ...],
    *,
    output: str | Path,
    manifest_output: str | Path,
) -> dict:
    if not tasks or any(task.split != "train" for task in tasks):
        raise ValueError("V1 grounded SFT export accepts Train tasks only")
    selected = tuple(
        task
        for task in tasks
        if task.provenance["archetype"]
        in {
            "shared_impact_accept",
            "shared_impact_decline",
            "dropped_write_recovery",
        }
    )
    if not selected:
        raise ValueError("V1 grounded curriculum has no selected tasks")
    destination = Path(output)
    manifest_path = Path(manifest_output)
    for path in (destination, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite V1 grounded SFT artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    records = tuple(
        record
        for task in selected
        for record in _records_for_task(
            task,
            agent_task=_grounded_policy_task(task),
            normalize_idempotency_keys=True,
            format_version="deviceops-ul-sft-v1.2-public-grounded",
        )
    )
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    ).encode("utf-8")
    hidden_markers = {str(task.initial_world["secret_marker"]) for task in selected}
    if any(marker.encode("utf-8") in serialized for marker in hidden_markers):
        raise ValueError("hidden world marker leaked into V1 grounded SFT data")
    reference_audit = audit_public_references(records)
    if reference_audit["ungrounded_reference_count"]:
        raise ValueError(
            "V1 grounded curriculum contains tool references absent from its prompt"
        )
    task_payload = "".join(
        json.dumps(asdict(task), ensure_ascii=False, sort_keys=True) + "\n"
        for task in selected
    ).encode("utf-8")
    manifest = {
        "format_version": "deviceops-ul-sft-manifest-v1.2",
        "generator": "deviceops-ul-public-grounded-curriculum-v1.2",
        "record_count": len(records),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "source_task_count": len(selected),
        "source_task_sha256": hashlib.sha256(task_payload).hexdigest(),
        "selected_archetypes": [
            "dropped_write_recovery",
            "shared_impact_accept",
            "shared_impact_decline",
        ],
        "hidden_marker_count": len(hidden_markers),
        "hidden_marker_leaks": 0,
        "hidden_seed_in_idempotency_keys": 0,
        **reference_audit,
    }
    destination.write_bytes(serialized)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def audit_public_references(records: tuple[dict, ...] | list[dict]) -> dict:
    """Prove that entity/operation references in tool targets already occur in the prompt."""
    grounded = 0
    ungrounded: list[dict[str, Any]] = []
    for record in records:
        prompt_payload = json.dumps(
            record["prompt"], ensure_ascii=False, sort_keys=True
        )
        for call in record["completion"][0].get("tool_calls", []):
            raw_arguments = call.get("function", {}).get("arguments", "{}")
            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else raw_arguments
            )
            if not isinstance(arguments, dict):
                continue
            for field in ("device_id", "target_id", "operation_id"):
                value = arguments.get(field)
                if not isinstance(value, str):
                    continue
                if value in prompt_payload:
                    grounded += 1
                else:
                    ungrounded.append(
                        {
                            "task_id": record["metadata"]["task_id"],
                            "decision_index": record["metadata"]["decision_index"],
                            "field": field,
                            "value": value,
                        }
                    )
    return {
        "grounded_reference_count": grounded,
        "ungrounded_reference_count": len(ungrounded),
        "ungrounded_references": ungrounded,
    }


def export_oracle_sft(
    tasks: tuple[V1TaskSpec, ...],
    *,
    output: str | Path,
    manifest_output: str | Path,
) -> dict:
    if not tasks or any(task.split != "train" for task in tasks):
        raise ValueError("V1 SFT export accepts Train tasks only")
    destination = Path(output)
    manifest_path = Path(manifest_output)
    for path in (destination, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite V1 SFT artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    records = tuple(
        record for task in tasks for record in _records_for_task(task)
    )
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    ).encode("utf-8")
    hidden_markers = {str(task.initial_world["secret_marker"]) for task in tasks}
    if any(marker.encode("utf-8") in serialized for marker in hidden_markers):
        raise ValueError("hidden world marker leaked into V1 SFT data")
    destination.write_bytes(serialized)
    task_payload = "".join(
        json.dumps(asdict(task), ensure_ascii=False, sort_keys=True) + "\n" for task in tasks
    ).encode("utf-8")
    manifest = {
        "format_version": "deviceops-ul-sft-manifest-v1.1",
        "generator": "deviceops-ul-oracle-next-decision-v1.1",
        "record_count": len(records),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "source_task_count": len(tasks),
        "source_task_sha256": hashlib.sha256(task_payload).hexdigest(),
        "hidden_marker_count": len(hidden_markers),
        "hidden_marker_leaks": 0,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def audit_records(
    records: list[dict], *, tokenizer: Any, max_length: int
) -> dict:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    lengths: list[int] = []
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    prefix_mismatch_count = 0
    for record in records:
        if "prompt" not in record or "completion" not in record:
            raise ValueError("V1.1 audit requires prompt-completion records")
        if not isinstance(record["completion"], list) or len(record["completion"]) != 1:
            raise ValueError("V1.1 completion must contain exactly one assistant decision")
        prompt_rendered = tokenizer.apply_chat_template(
            record["prompt"],
            tools=record["tools"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_rendered = tokenizer.apply_chat_template(
            record["prompt"] + record["completion"],
            tools=record["tools"],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_ids = _extract_token_ids(prompt_rendered)
        full_ids = _extract_token_ids(full_rendered)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            prefix_mismatch_count += 1
        completion_length = max(0, len(full_ids) - len(prompt_ids))
        lengths.append(len(full_ids))
        prompt_lengths.append(len(prompt_ids))
        completion_lengths.append(completion_length)
    if not lengths:
        raise ValueError("V1 SFT audit requires at least one record")
    ordered = sorted(lengths)
    truncation_count = sum(length > max_length for length in lengths)

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    return {
        "format_version": "deviceops-ul-sft-token-audit-v1.1",
        "record_count": len(lengths),
        "max_length": max_length,
        "min_tokens": ordered[0],
        "p50_tokens": percentile(0.50),
        "p95_tokens": percentile(0.95),
        "max_tokens": ordered[-1],
        "prompt_token_count": sum(prompt_lengths),
        "completion_token_count": sum(completion_lengths),
        "min_completion_tokens": min(completion_lengths),
        "max_completion_tokens": max(completion_lengths),
        "records_with_zero_completion_tokens": sum(
            length == 0 for length in completion_lengths
        ),
        "prompt_prefix_mismatch_count": prefix_mismatch_count,
        "completion_only_loss_verified": (
            all(length > 0 for length in completion_lengths)
            and prefix_mismatch_count == 0
        ),
        "truncation_count": truncation_count,
        "truncation_rate": truncation_count / len(lengths),
    }


def _extract_token_ids(rendered: Any) -> list[int]:
    token_ids = rendered.get("input_ids") if isinstance(rendered, Mapping) else rendered
    if not isinstance(token_ids, list):
        raise ValueError("chat template did not return a token ID list")
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("unexpected batched token IDs in V1 SFT audit")
        token_ids = token_ids[0]
    if not all(isinstance(token, int) for token in token_ids):
        raise ValueError("chat template returned non-integer token IDs")
    return token_ids


def audit_sft_file(
    *,
    data_path: str | Path,
    model_path: str | Path,
    max_length: int,
    output_path: str | Path,
) -> dict:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("transformers is required for the V1 token audit") from exc
    source = Path(data_path)
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    audit = audit_records(records, tokenizer=tokenizer, max_length=max_length)
    audit.update(
        {
            "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model_path": Path(model_path).as_posix(),
        }
    )
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite V1 token audit: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return audit
