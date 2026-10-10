from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from device_agent_arena.domain.models import (
    DeviceState,
    FaultSpec,
    GoalSpec,
    TaskSpec,
)
from device_agent_arena.simulator.tools import TOOL_NAMES

CANONICAL_CATEGORIES = {
    "direct",
    "multi_step",
    "state_dependency",
    "missing_parameter",
    "irrelevant",
    "asr_noise",
    "tool_failure",
    "preset",
}
RL_CATEGORIES = {"rl_compositional", "rl_fault", "rl_long_horizon"}
ALLOWED_CATEGORIES = CANONICAL_CATEGORIES | RL_CATEGORIES


class TaskValidationError(ValueError):
    pass


TASK_ID_PATTERN = re.compile(r"^(train|dev|test)-[a-z_]+-\d{3}$")
VERSION_PATTERN = re.compile(r"^\d+\.\d+$")
STATE_FIELDS = {
    "connected",
    "calibrated",
    "orientation",
    "tracking",
    "mode",
    "target",
    "presets",
}
SNAPSHOT_FIELDS = {"orientation", "tracking", "mode", "target"}


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskValidationError(f"{field} must be an object")
    return value


def _require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TaskValidationError(f"{field} must be a list")
    items = tuple(_require_non_empty_string(item, field) for item in value)
    if len(items) != len(set(items)):
        raise TaskValidationError(f"{field} must not contain duplicates")
    return items


def _validate_state_values(data: dict[str, Any], field: str, *, snapshot: bool = False) -> None:
    allowed = SNAPSHOT_FIELDS if snapshot else STATE_FIELDS
    unknown = set(data) - allowed
    if unknown:
        raise TaskValidationError(f"unknown {field} fields: {sorted(unknown)}")
    for boolean_field in ("connected", "calibrated"):
        if boolean_field in data and not isinstance(data[boolean_field], bool):
            raise TaskValidationError(f"{field}.{boolean_field} must be boolean")
    if "orientation" in data and data["orientation"] not in {"portrait", "landscape"}:
        raise TaskValidationError(f"invalid {field}.orientation")
    if "tracking" in data and data["tracking"] not in {"off", "person", "object"}:
        raise TaskValidationError(f"invalid {field}.tracking")
    if "mode" in data and data["mode"] not in {"normal", "presentation", "follow"}:
        raise TaskValidationError(f"invalid {field}.mode")
    if "target" in data and data["target"] is not None:
        _require_non_empty_string(data["target"], f"{field}.target")


def _device_state(data: dict[str, Any]) -> DeviceState:
    _validate_state_values(data, "initial_state")
    presets = data.get("presets", {})
    if not isinstance(presets, dict):
        raise TaskValidationError("initial_state.presets must be an object")
    for name, snapshot_data in presets.items():
        _require_non_empty_string(name, "preset name")
        snapshot = _require_mapping(snapshot_data, f"preset {name}")
        _validate_state_values(snapshot, f"preset {name}", snapshot=True)
    state = DeviceState(**data)
    if state.calibrated and not state.connected:
        raise TaskValidationError("initial_state cannot be calibrated while disconnected")
    if state.mode == "follow" and (not state.connected or not state.calibrated):
        raise TaskValidationError(
            "initial_state follow mode requires a connected and calibrated device"
        )
    if state.tracking == "off" and state.target is not None:
        raise TaskValidationError("initial_state.target must be null when tracking is off")
    if state.tracking != "off" and not state.target:
        raise TaskValidationError("initial_state.target is required when tracking is enabled")
    if state.tracking != "off" and (not state.connected or not state.calibrated):
        raise TaskValidationError(
            "initial_state active tracking requires a connected and calibrated device"
        )
    return state


def task_from_dict(data: dict[str, Any]) -> TaskSpec:
    expected_fields = {
        "id",
        "version",
        "split",
        "categories",
        "seed",
        "instruction",
        "reference_instruction",
        "initial_state",
        "fault_plan",
        "clarifications",
        "goal",
        "provenance",
    }
    unknown_fields = set(data) - expected_fields
    if unknown_fields:
        raise TaskValidationError(f"unknown task fields: {sorted(unknown_fields)}")
    task_id = _require_non_empty_string(data.get("id"), "id")
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise TaskValidationError("id must match <split>-<category>-<three digits>")
    version = _require_non_empty_string(data.get("version"), "version")
    if not VERSION_PATTERN.fullmatch(version):
        raise TaskValidationError(f"{task_id}: version must use <major>.<minor>")
    split = data.get("split")
    if split not in {"train", "dev", "test"}:
        raise TaskValidationError(f"{task_id}: split must be train, dev, or test")
    if not task_id.startswith(f"{split}-"):
        raise TaskValidationError(f"{task_id}: id prefix must match split")

    categories_data = data.get("categories")
    if not isinstance(categories_data, list) or not categories_data:
        raise TaskValidationError(f"{task_id}: categories must be a non-empty list")
    categories = _string_list(categories_data, "categories")
    unknown_categories = set(categories) - ALLOWED_CATEGORIES
    if unknown_categories:
        raise TaskValidationError(f"{task_id}: unknown categories {sorted(unknown_categories)}")

    seed = data.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TaskValidationError(f"{task_id}: seed must be an integer")

    initial_state = _device_state(_require_mapping(data.get("initial_state"), "initial_state"))

    fault_data = data.get("fault_plan", [])
    if not isinstance(fault_data, list):
        raise TaskValidationError(f"{task_id}: fault_plan must be a list")
    faults: list[FaultSpec] = []
    for item in fault_data:
        fault = _require_mapping(item, "fault_plan item")
        unknown_fault_fields = set(fault) - {"tool", "invocation", "code", "probability"}
        if unknown_fault_fields:
            raise TaskValidationError(
                f"{task_id}: unknown fault fields {sorted(unknown_fault_fields)}"
            )
        tool = _require_non_empty_string(fault.get("tool"), "fault tool")
        if tool not in TOOL_NAMES:
            raise TaskValidationError(f"{task_id}: unknown fault tool {tool}")
        invocation = fault.get("invocation", 1)
        if not isinstance(invocation, int) or isinstance(invocation, bool) or invocation < 1:
            raise TaskValidationError(f"{task_id}: fault invocation must be >= 1")
        code = _require_non_empty_string(fault.get("code", "transient_failure"), "fault code")
        probability = fault.get("probability", 1.0)
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not 0.0 < float(probability) <= 1.0
        ):
            raise TaskValidationError(f"{task_id}: fault probability must be in (0, 1]")
        faults.append(
            FaultSpec(
                tool=tool,
                invocation=invocation,
                code=code,
                probability=float(probability),
            )
        )

    clarification_data = _require_mapping(data.get("clarifications", {}), "clarifications")
    clarifications = {
        _require_non_empty_string(key, "clarification key"): _require_non_empty_string(
            value, "clarification value"
        )
        for key, value in clarification_data.items()
    }

    goal_data = _require_mapping(data.get("goal"), "goal")
    unknown_goal_fields = set(goal_data) - {
        "final_state",
        "required_tools",
        "forbidden_tools",
        "required_clarifications",
        "terminal_behavior",
    }
    if unknown_goal_fields:
        raise TaskValidationError(f"{task_id}: unknown goal fields {sorted(unknown_goal_fields)}")
    terminal_behavior = goal_data.get("terminal_behavior", "answer")
    if terminal_behavior not in {"answer", "refuse"}:
        raise TaskValidationError(f"{task_id}: invalid terminal_behavior")
    final_state = _require_mapping(goal_data.get("final_state", {}), "goal.final_state")
    _validate_state_values(final_state, "goal.final_state")
    required_tools = _string_list(goal_data.get("required_tools", []), "goal.required_tools")
    forbidden_tools = _string_list(goal_data.get("forbidden_tools", []), "goal.forbidden_tools")
    unknown_tools = (set(required_tools) | set(forbidden_tools)) - TOOL_NAMES
    if unknown_tools:
        raise TaskValidationError(f"{task_id}: unknown goal tools {sorted(unknown_tools)}")
    if set(required_tools) & set(forbidden_tools):
        raise TaskValidationError(f"{task_id}: a tool cannot be both required and forbidden")
    required_clarifications = _string_list(
        goal_data.get("required_clarifications", []), "goal.required_clarifications"
    )
    missing_scripts = set(required_clarifications) - set(clarifications)
    if missing_scripts:
        raise TaskValidationError(
            f"{task_id}: missing scripted clarifications {sorted(missing_scripts)}"
        )
    goal = GoalSpec(
        final_state=final_state,
        required_tools=required_tools,
        forbidden_tools=forbidden_tools,
        required_clarifications=required_clarifications,
        terminal_behavior=terminal_behavior,
    )

    provenance_data = _require_mapping(data.get("provenance"), "provenance")
    provenance = {
        _require_non_empty_string(key, "provenance key"): _require_non_empty_string(
            value, "provenance value"
        )
        for key, value in provenance_data.items()
    }

    return TaskSpec(
        id=task_id,
        version=version,
        split=split,
        categories=categories,
        seed=seed,
        instruction=_require_non_empty_string(data.get("instruction"), "instruction"),
        reference_instruction=_require_non_empty_string(
            data.get("reference_instruction"), "reference_instruction"
        ),
        initial_state=initial_state,
        fault_plan=tuple(faults),
        clarifications=clarifications,
        goal=goal,
        provenance=provenance,
    )


def load_tasks(path: str | Path) -> tuple[TaskSpec, ...]:
    source = Path(path)
    tasks: list[TaskSpec] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskValidationError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
        task = task_from_dict(_require_mapping(raw, "task"))
        if task.id in seen:
            raise TaskValidationError(f"duplicate task id: {task.id}")
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise TaskValidationError(f"{source}: no tasks found")
    return tuple(tasks)


def write_tasks(path: str | Path, tasks: tuple[TaskSpec, ...] | list[TaskSpec]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(task), ensure_ascii=False, sort_keys=True) for task in tasks]
    destination.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
