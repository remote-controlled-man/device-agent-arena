from __future__ import annotations

import re
from collections import deque
from copy import deepcopy

from device_agent_arena.domain.models import ToolCall, TrajectoryEvent
from device_agent_arena.v1.catalog import public_name_for_operation
from device_agent_arena.v1.models import (
    DialogueAction,
    V1Observation,
    V1TaskSpec,
)


class OracleAgentV1:
    """Hidden-state solvability oracle; never use as a deployable baseline."""

    name = "oracle-v1.0"

    def __init__(self, task: V1TaskSpec) -> None:
        self._profile = task.tool_profile
        self._plan = deque(task.oracle_plan)

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction:
        if not self._plan:
            return DialogueAction("answer", "任务已完成。")
        step = self._plan.popleft()
        if step.kind == "tool":
            arguments = _resolve_placeholders(step.arguments, observation.events)
            return ToolCall(
                public_name_for_operation(self._profile, step.name),
                arguments,
            )
        return DialogueAction(
            kind=step.kind,  # type: ignore[arg-type]
            content=step.content,
            field=step.field,
            operation=step.operation,
        )


class AlwaysAskAgent:
    name = "always-ask-v1.0"

    def __init__(self) -> None:
        self._asked = False

    def act(self, observation: V1Observation) -> DialogueAction:
        if not self._asked:
            self._asked = True
            return DialogueAction("clarify", "请再明确一下目标设备。", field="device")
        return DialogueAction("refuse", "信息仍不足，我没有执行任何变更。")


class NoToolAgent:
    name = "no-tool-v1.0"

    def act(self, observation: V1Observation) -> DialogueAction:
        return DialogueAction("answer", "已经处理完成。")


class DirectExecutionAgent:
    """Ablation: use the reactive policy but skip all requested interaction."""

    name = "direct-execution-v1.0"

    def __init__(self) -> None:
        self._delegate = NaiveHeuristicAgent()

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction:
        action = self._delegate.act(observation)
        if isinstance(action, DialogueAction) and action.kind in {"clarify", "confirm"}:
            return DialogueAction("answer", "未等待用户交互，直接结束。")
        return action


class NaiveHeuristicAgent:
    """Reactive public-observation baseline that intentionally has no hidden-task access."""

    name = "naive-heuristic-v1.0"

    def __init__(self) -> None:
        self._profile: str | None = None
        self._target: str | None = None
        self._search_matches: list[dict] = []
        self._pending: deque[tuple[str, dict]] = deque()
        self._retry: tuple[str, dict] | None = None
        self._key_index = 0
        self._initialized = False

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction:
        if self._profile is None:
            self._profile = _infer_profile(observation)
        recent = _events_since_last_decision(observation.events)
        user = next((event for event in reversed(recent) if event.kind == "user"), None)
        if user is not None:
            if user.payload.get("field") == "device":
                label = str(user.payload.get("content", ""))
                match = next(
                    (item for item in self._search_matches if item.get("label") == label),
                    None,
                )
                if match:
                    self._target = str(match["device_id"])
                if self._retry is not None:
                    operation, arguments = self._retry
                    self._retry = None
                    arguments["target_id"] = self._target
                    return self._call(operation, arguments)
            if user.payload.get("operation"):
                if user.payload.get("content") != "yes":
                    return DialogueAction(
                        "refuse", "你没有确认会影响他人的变更，因此我没有继续执行。"
                    )
                if self._retry is not None:
                    operation, arguments = self._retry
                    self._retry = None
                    return self._call(operation, arguments)

        result = next((event for event in reversed(recent) if event.kind == "tool_result"), None)
        if result is not None:
            data = result.payload.get("data", {})
            code = result.payload.get("code")
            if code == "clarification_required" and self._retry is not None:
                return DialogueAction("clarify", "你具体指的是哪一个设备？", field="device")
            if code == "confirmation_required" and self._retry is not None:
                operation = result.payload.get("details", {}).get("confirmation_key")
                return DialogueAction(
                    "confirm",
                    "该操作会影响共享设备或他人的自动化，是否继续？",
                    operation=str(operation),
                )
            if isinstance(data, dict) and "matches" in data:
                self._search_matches = list(data["matches"])
                if not self._search_matches:
                    return DialogueAction("refuse", "没有找到匹配设备。")
                self._target = str(self._search_matches[0]["device_id"])
                return self._call("inspect_device", {"device_id": self._target})
            if isinstance(data, dict) and data.get("id") == self._target:
                if "小林" in observation.instruction or "定时" in observation.instruction:
                    return self._call("inspect_routines", {"device_id": self._target})
                self._queue_device_changes(observation.instruction)
                return self._next_or_answer()
            if isinstance(data, dict) and "routines" in data:
                active = next(
                    (item for item in data["routines"] if item.get("active")),
                    None,
                )
                if active:
                    self._pending.append(
                        self._mutation("pause_routine", str(active["id"]), {})
                    )
                self._pending.append(
                    self._mutation("set_mode", str(self._target), {"mode": "cinema"})
                )
                return self._next_or_answer()
            if isinstance(data, dict) and "operation_id" in data and code in {
                "accepted",
                "timeout",
                "idempotent_replay",
            }:
                return self._call("poll", {"operation_id": data["operation_id"]})
            if isinstance(data, dict) and "status" in data:
                if data["status"] == "failed":
                    return DialogueAction("answer", "操作返回失败，我先结束本次任务。")
                return self._next_or_answer()

        if not self._initialized:
            self._initialized = True
            explicit = re.search(r"设备\s+([\w-]+)", observation.instruction)
            if explicit:
                self._target = explicit.group(1)
                return self._call("inspect_device", {"device_id": self._target})
            query = "客厅" if "客厅" in observation.instruction else "卧室灯"
            return self._call("search", {"query": query})
        return self._next_or_answer()

    def _queue_device_changes(self, instruction: str) -> None:
        if self._pending or self._target is None:
            return
        mode = "normal"
        for word, value in (
            ("睡眠", "sleep"),
            ("观影", "cinema"),
            ("专注", "focus"),
            ("阅读", "reading"),
        ):
            if word in instruction:
                mode = value
                break
        self._pending.append(self._mutation("set_mode", self._target, {"mode": mode}))
        if "7 点" in instruction or "7点" in instruction:
            self._pending.append(
                self._mutation(
                    "schedule_restore",
                    self._target,
                    {"time": "07:00", "mode": "normal"},
                )
            )

    def _mutation(self, action: str, target_id: str, payload: dict) -> tuple[str, dict]:
        self._key_index += 1
        return (
            "apply_change",
            {
                "action": action,
                "target_id": target_id,
                "payload": payload,
                "idempotency_key": f"heuristic-{self._key_index}",
            },
        )

    def _next_or_answer(self) -> ToolCall | DialogueAction:
        if not self._pending:
            return DialogueAction("answer", "任务已经处理完成。")
        operation, arguments = self._pending.popleft()
        if operation == "apply_change":
            self._retry = (operation, deepcopy(arguments))
        return self._call(operation, arguments)

    def _call(self, operation: str, arguments: dict) -> ToolCall:
        assert self._profile is not None
        return ToolCall(public_name_for_operation(self._profile, operation), arguments)


def _resolve_placeholders(
    arguments: dict, events: tuple[TrajectoryEvent, ...]
) -> dict:
    resolved = deepcopy(arguments)
    if resolved.get("operation_id") == "$last_operation":
        operation_id = next(
            (
                event.payload.get("data", {}).get("operation_id")
                for event in reversed(events)
                if event.kind == "tool_result"
                and event.payload.get("data", {}).get("operation_id")
            ),
            None,
        )
        if operation_id is None:
            raise RuntimeError("oracle plan expected a previous operation id")
        resolved["operation_id"] = operation_id
    return resolved


def _infer_profile(observation: V1Observation) -> str:
    names = {tool.name for tool in observation.tools}
    if "search_devices" in names:
        return "home-a"
    if "find_endpoints" in names:
        return "home-b"
    if "locate_nodes" in names:
        return "home-x"
    raise ValueError("unknown public tool profile")


def _events_since_last_decision(
    events: tuple[TrajectoryEvent, ...]
) -> tuple[TrajectoryEvent, ...]:
    if not events:
        return ()
    latest_step = max(event.step for event in events)
    return tuple(event for event in events if event.step == latest_step)
