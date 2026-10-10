from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from device_agent_arena.domain.models import DeviceState, FaultSpec, ToolCall, ToolResult
from device_agent_arena.prompts import BASE_SYSTEM_PROMPT
from device_agent_arena.simulator.device import DeviceSimulator
from device_agent_arena.simulator.tools import (
    INTERACTION_TOOL_NAMES,
    TOOL_NAMES,
    tool_schemas,
    validate_tool_arguments,
)
from device_agent_arena.training.data import convert_sft_records_for_functiongemma

STATE_LATTICE_ACTIONS = (
    "request_clarification",
    "get_device_state",
    "connect_device",
    "calibrate_device",
    "recall_preset",
    "set_mode",
    "set_orientation",
    "set_tracking",
    "save_preset",
    "finish_task",
)
HELDOUT_MARKERS = ("海棠", "松石", "联调", "巡演")
_GENERATOR = "functiongemma-state-lattice-v0.5"
_CURRICULUM_GENERATOR = "functiongemma-state-lattice-curriculum-v0.5"


def _assistant_call(name: str, arguments: dict[str, Any], call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                },
            }
        ],
    }


def _tool_result_message(call: ToolCall, result: ToolResult, call_id: str) -> dict:
    return {
        "role": "tool",
        "name": call.name,
        "tool_call_id": call_id,
        "content": json.dumps(
            {
                "name": call.name,
                "ok": result.ok,
                "code": result.code,
                "message": result.message,
                "state_delta": result.state_delta,
                "data": result.data,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _append_simulated_call(
    messages: list[dict], simulator: DeviceSimulator, call: ToolCall, call_id: str
) -> ToolResult:
    messages.append(_assistant_call(call.name, call.arguments, call_id))
    result = simulator.execute(call)
    messages.append(_tool_result_message(call, result, call_id))
    return result


def _opposite_orientation(orientation: str) -> str:
    return "portrait" if orientation == "landscape" else "landscape"


def _case_values(action: str, index: int) -> dict[str, Any]:
    serial = STATE_LATTICE_ACTIONS.index(action) * 1000 + index + 1
    kind = "person" if index % 2 == 0 else "object"
    kind_word = "人物" if kind == "person" else "物体"
    target = f"云杉晶格{'成员' if kind == 'person' else '道具'}{serial:04d}"
    old_target = f"云杉晶格旧目标{serial:04d}"
    preset = f"云杉晶格方案{serial:04d}"
    orientation = "landscape" if index % 2 == 0 else "portrait"
    orientation_word = "横屏" if orientation == "landscape" else "竖屏"
    mode = "follow" if index % 3 else "presentation"
    desired_tracking = kind if mode == "follow" else "off"
    desired_target = target if desired_tracking != "off" else None
    snapshot = {
        "orientation": orientation,
        "tracking": desired_tracking,
        "mode": mode,
        "target": desired_target,
    }
    return {
        "serial": serial,
        "kind": kind,
        "kind_word": kind_word,
        "target": target,
        "old_target": old_target,
        "preset": preset,
        "orientation": orientation,
        "orientation_word": orientation_word,
        "mode": mode,
        "desired_tracking": desired_tracking,
        "desired_target": desired_target,
        "snapshot": snapshot,
    }


def _instruction(action: str, values: dict[str, Any]) -> str:
    scene = values["serial"]
    target = values["target"]
    preset = values["preset"]
    orientation_word = values["orientation_word"]
    kind_word = values["kind_word"]
    if action == "request_clarification":
        return (
            f"设备已在线并完成校准。请开启{kind_word}跟踪，但我还没有说明具体目标；"
            f"缺少目标时先询问，不要猜。训练场景{scene:04d}。"
        )
    if action == "get_device_state":
        return (
            f"先读取当前设备状态，再继续调整为{orientation_word}并跟踪{target}；"
            f"本轮只调用一个必要工具。训练场景{scene:04d}。"
        )
    if action == "connect_device":
        return (
            f"设备当前未连接，请继续完成{orientation_word}配置并以{kind_word}方式跟踪"
            f"{target}；先满足前置条件，每轮只调用一个工具。训练场景{scene:04d}。"
        )
    if action == "calibrate_device":
        return (
            f"设备已连接但尚未校准，请继续进入跟随模式并跟踪{target}；"
            f"先满足前置条件，每轮只调用一个工具。训练场景{scene:04d}。"
        )
    if action == "recall_preset":
        return (
            f"设备已连接并校准，请恢复预设“{preset}”；不要重做预设中已有的设置。"
            f"训练场景{scene:04d}。"
        )
    if action == "set_mode":
        mode_word = "跟随" if values["mode"] == "follow" else "演示"
        return (
            f"连接和校准均已完成，当前其他设置无需改动；请切换到{mode_word}模式。"
            f"训练场景{scene:04d}。"
        )
    if action == "set_orientation":
        return (
            f"设备已连接并校准，模式与跟踪目标已经正确，只需调整为{orientation_word}。"
            f"训练场景{scene:04d}。"
        )
    if action == "set_tracking":
        return (
            f"设备已连接并校准，方向和模式已经正确；现在以{kind_word}方式跟踪{target}。"
            f"训练场景{scene:04d}。"
        )
    if action == "save_preset":
        return (
            f"设备已达到{orientation_word}跟随{target}的目标状态，最后还要保存为“{preset}”；"
            f"保存前不要提前结束。训练场景{scene:04d}。"
        )
    if action == "finish_task":
        return (
            f"请把设备调成{orientation_word}跟随{target}并保存为“{preset}”。"
            f"若所有设置和保存均已完成，就结束任务，不要重复调用。训练场景{scene:04d}。"
        )
    raise AssertionError(f"unsupported state-lattice action: {action}")


def _state_for_action(action: str, values: dict[str, Any]) -> DeviceState:
    orientation = values["orientation"]
    mode = values["mode"]
    target = values["target"]
    kind = values["kind"]
    preset = values["preset"]
    snapshot = values["snapshot"]
    if action in {"connect_device", "get_device_state"}:
        return DeviceState(orientation=_opposite_orientation(orientation))
    if action == "request_clarification":
        return DeviceState(connected=True, calibrated=True, mode="follow")
    if action == "calibrate_device":
        return DeviceState(
            connected=True,
            calibrated=False,
            orientation=orientation,
            mode="normal",
        )
    if action == "recall_preset":
        return DeviceState(
            connected=True,
            calibrated=True,
            orientation=_opposite_orientation(orientation),
            mode="normal",
            presets={preset: deepcopy(snapshot)},
        )
    if action == "set_mode":
        current_mode = "normal" if mode != "normal" else "presentation"
        return DeviceState(
            connected=True,
            calibrated=True,
            orientation=orientation,
            tracking=values["desired_tracking"],
            mode=current_mode,
            target=values["desired_target"],
        )
    if action == "set_orientation":
        return DeviceState(
            connected=True,
            calibrated=True,
            orientation=_opposite_orientation(orientation),
            tracking=values["desired_tracking"],
            mode=mode,
            target=values["desired_target"],
        )
    if action == "set_tracking":
        return DeviceState(
            connected=True,
            calibrated=True,
            orientation=orientation,
            tracking="off" if kind != "off" else kind,
            mode="follow",
            target=None,
        )
    if action == "save_preset":
        return DeviceState(
            connected=True,
            calibrated=True,
            orientation=orientation,
            tracking=kind,
            mode="follow",
            target=target,
        )
    if action == "finish_task":
        return DeviceState(
            connected=True,
            calibrated=True,
            orientation=orientation,
            tracking=kind,
            mode="follow",
            target=target,
            presets={
                preset: {
                    "orientation": orientation,
                    "tracking": kind,
                    "mode": "follow",
                    "target": target,
                }
            },
        )
    raise AssertionError(f"unsupported state-lattice action: {action}")


def _arguments_for_action(action: str, values: dict[str, Any]) -> dict[str, Any]:
    if action == "request_clarification":
        return {"field": "target", "content": "请提供需要跟踪的具体目标。"}
    if action in {"get_device_state", "connect_device", "calibrate_device"}:
        return {}
    if action == "recall_preset":
        return {"name": values["preset"]}
    if action == "set_mode":
        return {"mode": values["mode"]}
    if action == "set_orientation":
        return {"orientation": values["orientation"]}
    if action == "set_tracking":
        return {"kind": values["kind"], "target": values["target"]}
    if action == "save_preset":
        return {"name": values["preset"]}
    if action == "finish_task":
        return {"kind": "answer", "content": "设备配置和预设保存均已完成。"}
    raise AssertionError(f"unsupported state-lattice action: {action}")


def _append_context(
    action: str,
    index: int,
    messages: list[dict],
    simulator: DeviceSimulator,
    values: dict[str, Any],
) -> str:
    if action in {"request_clarification", "get_device_state"}:
        return "instruction_only"
    if action == "connect_device" and index % 2:
        attempted = ToolCall(
            "set_orientation", {"orientation": values["orientation"]}
        )
        result = _append_simulated_call(messages, simulator, attempted, "context_1")
        if result.ok or result.code != "precondition_failed":
            raise AssertionError("disconnected recovery context must fail its dependent action")
        return "precondition_failure"
    if action == "calibrate_device" and index % 2:
        attempted = ToolCall(
            "set_tracking", {"kind": values["kind"], "target": values["target"]}
        )
        result = _append_simulated_call(messages, simulator, attempted, "context_1")
        if result.ok or result.code != "precondition_failed":
            raise AssertionError("uncalibrated recovery context must fail active tracking")
        return "precondition_failure"
    if action in {"recall_preset", "set_mode", "set_orientation", "set_tracking"} and (
        index % 4 == 3
    ):
        arguments = _arguments_for_action(action, values)
        failed_simulator = DeviceSimulator(
            deepcopy(simulator.state),
            faults=(
                FaultSpec(
                    tool=action,
                    invocation=1,
                    code="transient_failure",
                    probability=1.0,
                ),
            ),
            seed=index,
        )
        result = _append_simulated_call(
            messages,
            failed_simulator,
            ToolCall(action, arguments),
            "context_1",
        )
        if result.ok or result.code != "transient_failure":
            raise AssertionError("retry context must contain a transient tool failure")
        return "transient_retry"
    _append_simulated_call(
        messages,
        simulator,
        ToolCall("get_device_state"),
        "context_1",
    )
    return "state_snapshot"


def _generate_openai_state_lattice_records(*, per_action: int) -> list[dict]:
    if per_action < 1 or per_action > 256:
        raise ValueError("per_action must be between 1 and 256")
    schemas = tool_schemas(include_interaction=True)
    records: list[dict] = []
    for action in STATE_LATTICE_ACTIONS:
        for index in range(per_action):
            values = _case_values(action, index)
            messages = [
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user", "content": _instruction(action, values)},
            ]
            simulator = DeviceSimulator(_state_for_action(action, values), seed=index)
            context_style = _append_context(
                action,
                index,
                messages,
                simulator,
                values,
            )
            arguments = _arguments_for_action(action, values)
            completion = _assistant_call(action, arguments, "decision_1")
            canonical = json.dumps(
                {"prompt": messages, "completion": [completion]},
                ensure_ascii=False,
                sort_keys=True,
            )
            records.append(
                {
                    "prompt": messages,
                    "completion": [completion],
                    "tools": schemas,
                    "metadata": {
                        "category": "state_lattice",
                        "source_split": "train",
                        "source_index": index,
                        "case_id": f"{action}-{index:03d}",
                        "next_action": action,
                        "context_style": context_style,
                        "state_before": asdict(simulator.state),
                        "decision_type": "tool_call",
                        "generator": _GENERATOR,
                        "canonical_hash": hashlib.sha256(
                            canonical.encode("utf-8")
                        ).hexdigest(),
                    },
                }
            )
    return records


def generate_functiongemma_state_lattice_records(*, per_action: int = 32) -> list[dict]:
    """Generate balanced train-only local state/action supervision in native protocol."""
    return convert_sft_records_for_functiongemma(
        _generate_openai_state_lattice_records(per_action=per_action)
    )


def build_functiongemma_state_lattice_curriculum_v05(
    structural_records: Iterable[dict],
    *,
    per_action: int = 32,
) -> list[dict]:
    """Mix unique Train128 replay with local state/action lattice supervision."""
    mixed: list[dict] = []
    seen_replay_hashes: set[str] = set()
    for source in structural_records:
        if source.get("metadata", {}).get("source_split") != "train":
            raise ValueError("state-lattice curriculum accepts train-split replay only")
        if "oversample_index" in source.get("metadata", {}):
            continue
        source_hash = str(source["metadata"]["canonical_hash"])
        if source_hash in seen_replay_hashes:
            continue
        seen_replay_hashes.add(source_hash)
        record = deepcopy(source)
        record["metadata"]["generator"] = _CURRICULUM_GENERATOR
        record["metadata"]["curriculum_axis"] = "structural_replay"
        mixed.append(record)

    if not mixed:
        raise ValueError("at least one unique structural replay record is required")
    lattice = _generate_openai_state_lattice_records(per_action=per_action)
    for record in lattice:
        record["metadata"]["generator"] = _CURRICULUM_GENERATOR
        record["metadata"]["curriculum_axis"] = "state_lattice"
        mixed.append(record)

    converted = convert_sft_records_for_functiongemma(mixed)
    hashes = [record["metadata"]["canonical_hash"] for record in converted]
    if len(set(hashes)) != len(hashes):
        raise ValueError("state-lattice curriculum contains duplicate supervision")
    return converted


def audit_functiongemma_state_lattice_records(records: Iterable[dict]) -> dict[str, Any]:
    """Verify native formatting, split hygiene, schema validity, and executability."""
    materialized = list(records)
    if not materialized:
        raise ValueError("at least one state-lattice record is required")
    action_counts: Counter[str] = Counter()
    device_calls = 0
    successful_device_calls = 0
    schema_errors: list[dict[str, str]] = []
    protocol_errors: list[str] = []

    for record in materialized:
        metadata = record.get("metadata", {})
        completion = record.get("completion", [{}])[0]
        calls = completion.get("tool_calls", [])
        case_id = str(metadata.get("case_id", "unknown"))
        if not record.get("prompt") or record["prompt"][0].get("role") != "developer":
            protocol_errors.append(case_id)
        if len(record.get("tools", [])) != 10 or len(calls) != 1:
            protocol_errors.append(case_id)
            continue
        function = calls[0].get("function", {})
        action = str(function.get("name", ""))
        arguments = function.get("arguments")
        action_counts[action] += 1
        if action != metadata.get("next_action"):
            protocol_errors.append(case_id)
        validation_error = validate_tool_arguments(action, arguments)
        if validation_error is not None:
            schema_errors.append({"case_id": case_id, "error": validation_error})
            continue
        if action in TOOL_NAMES:
            device_calls += 1
            state = DeviceState(**deepcopy(metadata["state_before"]))
            result = DeviceSimulator(state).execute(ToolCall(action, deepcopy(arguments)))
            successful_device_calls += int(result.ok)
        elif action not in INTERACTION_TOOL_NAMES:
            schema_errors.append({"case_id": case_id, "error": "unknown completion tool"})

    serialized = json.dumps(materialized, ensure_ascii=False, sort_keys=True)
    heldout_matches = [marker for marker in HELDOUT_MARKERS if marker in serialized]
    hashes = [str(record.get("metadata", {}).get("canonical_hash", "")) for record in materialized]
    unique_count = len(set(hashes))
    success_rate = successful_device_calls / device_calls if device_calls else 0.0
    expected_actions = set(STATE_LATTICE_ACTIONS)
    verified = (
        not protocol_errors
        and not schema_errors
        and not heldout_matches
        and unique_count == len(materialized)
        and set(action_counts) == expected_actions
        and success_rate == 1.0
    )
    return {
        "status": "verified" if verified else "invalid",
        "generator": materialized[0]["metadata"].get("generator"),
        "record_count": len(materialized),
        "unique_supervision_count": unique_count,
        "next_action_counts": dict(sorted(action_counts.items())),
        "device_call_count": device_calls,
        "device_call_success_rate": success_rate,
        "heldout_marker_matches": heldout_matches,
        "schema_errors": schema_errors,
        "protocol_error_case_ids": sorted(set(protocol_errors)),
    }
