from __future__ import annotations

from copy import deepcopy
from typing import Any

from device_agent_arena.domain.models import ToolCall, TrajectoryEvent
from device_agent_arena.v1.models import DialogueAction, PublicTool, V1Observation
from device_agent_arena.v1.runner import AgentV1


class SharedImpactGuardV2:
    """Enforce shared-impact preflight using public observations only."""

    def __init__(self, delegate: AgentV1) -> None:
        self.delegate = delegate
        self._processed_events = 0
        self._devices: dict[str, dict[str, Any]] = {}
        self._routines_by_device: dict[str, list[dict[str, Any]]] = {}
        self._confirmed: set[str] = set()
        self._declined: set[str] = set()
        self._pending_pause: dict[str, str] = {}
        self._operation_to_routine: dict[str, str] = {}
        self._paused: set[str] = set()
        self._telemetry: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return f"shared-impact-guard:{self.delegate.name}"

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction:
        self._ingest(observation.events)
        proposed = self.delegate.act(observation)
        drain = getattr(self.delegate, "drain_telemetry", None)
        delegate_telemetry = drain() if callable(drain) else None
        self._telemetry = dict(delegate_telemetry or {})
        if not isinstance(proposed, ToolCall):
            return proposed
        arguments = proposed.arguments
        action = arguments.get("action")
        target_id = arguments.get("target_id")
        if action not in {"set_mode", "schedule_restore", "wake_device", "pause_routine"}:
            return proposed
        if not isinstance(target_id, str):
            return proposed
        if action == "pause_routine":
            return self._guard_pause(proposed, target_id)
        if target_id not in self._devices:
            inspect = _find_public_tool(
                observation.tools,
                required=("device_id",),
                description_any=("当前状态", "device state"),
            )
            if inspect is not None:
                return self._decision(
                    ToolCall(inspect.name, {"device_id": target_id}),
                    "inspect_device_before_mutation",
                )
            return proposed
        if not bool(self._devices[target_id].get("shared")):
            return proposed
        routines = self._routines_by_device.get(target_id)
        if routines is None:
            inspect = _find_public_tool(
                observation.tools,
                required=("device_id",),
                description_any=("自动化", "routine", "flow"),
            )
            if inspect is not None:
                return self._decision(
                    ToolCall(inspect.name, {"device_id": target_id}),
                    "inspect_shared_dependencies",
                )
            return proposed
        active = [
            routine
            for routine in routines
            if bool(routine.get("active")) and str(routine.get("id")) not in self._paused
        ]
        if not active:
            return proposed
        routine = active[0]
        routine_id = str(routine["id"])
        confirmation = f"pause_routine:{routine_id}"
        if confirmation in self._declined:
            return self._decision(
                DialogueAction(
                    "refuse",
                    "你没有同意暂停共享自动化，因此没有执行设备变更。",
                ),
                "declined_shared_impact",
            )
        if confirmation not in self._confirmed:
            owner = str(routine.get("owner", "他人"))
            return self._decision(
                DialogueAction(
                    "confirm",
                    f"这会暂停{owner}的共享自动化 {routine_id}，是否继续？",
                    operation=confirmation,
                ),
                "confirm_shared_impact",
            )
        pending_operation = self._pending_pause.get(routine_id)
        if pending_operation:
            poll = _find_public_tool(
                observation.tools,
                required=("operation_id",),
                description_any=("异步操作", "operation", "job"),
            )
            if poll is not None:
                return self._decision(
                    ToolCall(poll.name, {"operation_id": pending_operation}),
                    "verify_pause_before_device_mutation",
                )
        pause = ToolCall(
            proposed.name,
            {"action": "pause_routine", "target_id": routine_id, "payload": {}},
        )
        return self._decision(pause, "pause_dependency_before_device_mutation")

    def drain_telemetry(self) -> dict[str, Any] | None:
        telemetry = self._telemetry
        self._telemetry = None
        return telemetry

    def _guard_pause(self, proposed: ToolCall, target_id: str) -> ToolCall | DialogueAction:
        routine = next(
            (
                item
                for routines in self._routines_by_device.values()
                for item in routines
                if str(item.get("id")) == target_id
            ),
            None,
        )
        if routine is None:
            return proposed
        confirmation = f"pause_routine:{target_id}"
        if confirmation in self._declined:
            return self._decision(
                DialogueAction("refuse", "你没有同意暂停共享自动化。"),
                "declined_shared_impact",
            )
        if confirmation not in self._confirmed:
            return self._decision(
                DialogueAction(
                    "confirm",
                    f"这会暂停{routine.get('owner', '他人')}的共享自动化 {target_id}，是否继续？",
                    operation=confirmation,
                ),
                "confirm_shared_impact",
            )
        return proposed

    def _decision(
        self, action: ToolCall | DialogueAction, reason: str
    ) -> ToolCall | DialogueAction:
        assert self._telemetry is not None
        self._telemetry["shared_impact_guard"] = {"intervened": True, "reason": reason}
        return action

    def _ingest(self, events: tuple[TrajectoryEvent, ...]) -> None:
        new_events = events[self._processed_events :]
        self._processed_events = len(events)
        by_step: dict[int, list[TrajectoryEvent]] = {}
        for event in new_events:
            by_step.setdefault(event.step, []).append(event)
            if event.kind == "user":
                operation = event.payload.get("operation")
                if isinstance(operation, str):
                    if event.payload.get("content") == "yes":
                        self._confirmed.add(operation)
                    else:
                        self._declined.add(operation)
        for step_events in by_step.values():
            call = next((event for event in step_events if event.kind == "tool_call"), None)
            result = next((event for event in step_events if event.kind == "tool_result"), None)
            if call is None or result is None:
                continue
            arguments = call.payload.get("arguments", {})
            data = result.payload.get("data", {})
            if not isinstance(arguments, dict) or not isinstance(data, dict):
                continue
            if "id" in data and "shared" in data:
                self._devices[str(data["id"])] = deepcopy(data)
            if isinstance(data.get("routines"), list):
                device_id = arguments.get("device_id")
                if isinstance(device_id, str):
                    self._routines_by_device[device_id] = deepcopy(data["routines"])
            if arguments.get("action") == "pause_routine":
                routine_id = arguments.get("target_id")
                operation_id = data.get("operation_id")
                if isinstance(routine_id, str) and isinstance(operation_id, str):
                    self._pending_pause[routine_id] = operation_id
                    self._operation_to_routine[operation_id] = routine_id
            operation_id = arguments.get("operation_id")
            if isinstance(operation_id, str) and operation_id in self._operation_to_routine:
                routine_id = self._operation_to_routine[operation_id]
                if data.get("status") == "completed":
                    self._paused.add(routine_id)
                    self._pending_pause.pop(routine_id, None)
                elif data.get("status") == "failed":
                    self._pending_pause.pop(routine_id, None)


def _find_public_tool(
    tools: tuple[PublicTool, ...],
    *,
    required: tuple[str, ...],
    description_any: tuple[str, ...],
) -> PublicTool | None:
    for tool in tools:
        properties = tool.parameters.get("properties", {})
        if set(properties) != set(required):
            continue
        description = tool.description.lower()
        if any(token.lower() in description for token in description_any):
            return tool
    return None
