from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from device_agent_arena.domain.models import ToolCall, ToolResult, TrajectoryEvent
from device_agent_arena.harness.types import HarnessConfig
from device_agent_arena.v1.catalog import ToolAdapter, adapter_for_name
from device_agent_arena.v1.grading import grade_v1
from device_agent_arena.v1.models import (
    DialogueAction,
    V1Observation,
    V1TaskSpec,
    V1TrialResult,
)


@dataclass(frozen=True)
class V1EnvironmentStep:
    observation: V1Observation
    emitted_events: tuple[TrajectoryEvent, ...]
    done: bool
    termination: str | None = None
    tool_result: ToolResult | None = None


class PartialContractEnvironment:
    """Stateful DeviceOps environment; contracts block unsafe actions but never plan."""

    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig(max_steps=12)
        self._task: V1TaskSpec | None = None
        self._world: dict[str, Any] = {}
        self._events: list[TrajectoryEvent] = []
        self._audit: list[dict[str, Any]] = []
        self._policy_violations: list[str] = []
        self._clarified: set[str] = set()
        self._confirmed: set[str] = set()
        self._invocations: Counter[str] = Counter()
        self._operation_invocations: Counter[str] = Counter()
        self._idempotency: dict[str, str] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._step = 1
        self._operation_index = 0
        self._total_cost = 0.0
        self._consecutive_tool_errors = 0
        self._termination: str | None = None
        self._started = 0.0

    @property
    def world(self) -> dict[str, Any]:
        return deepcopy(self._world)

    def reset(self, task: V1TaskSpec) -> V1Observation:
        self._task = task
        self._world = deepcopy(task.initial_world)
        self._events = []
        self._audit = []
        self._policy_violations = []
        self._clarified = set()
        self._confirmed = set()
        self._invocations = Counter()
        self._operation_invocations = Counter()
        self._idempotency = {}
        self._snapshots = {}
        self._step = 1
        self._operation_index = 0
        self._total_cost = 0.0
        self._consecutive_tool_errors = 0
        self._termination = None
        self._started = perf_counter()
        return task.public_observation(step=1, events=())

    def step(
        self,
        action: ToolCall | DialogueAction,
        *,
        model_event: dict[str, object] | None = None,
    ) -> V1EnvironmentStep:
        task = self._require_task()
        if self._termination is not None:
            raise RuntimeError("episode is already terminated")
        if perf_counter() - self._started >= self.config.timeout_seconds:
            self._termination = "timeout"
            return V1EnvironmentStep(self._observation(), (), True, self._termination)

        model_events = (
            [TrajectoryEvent(step=self._step, kind="model", payload=dict(model_event))]
            if model_event
            else []
        )
        if isinstance(action, ToolCall):
            emitted, result = self._tool_step(task, action)
        elif isinstance(action, DialogueAction):
            emitted, result = self._dialogue_step(task, action), None
        else:
            raise TypeError(f"unsupported v1 action: {type(action).__name__}")

        emitted = [*model_events, *emitted]
        self._events.extend(emitted)
        if result is not None:
            if result.ok:
                self._consecutive_tool_errors = 0
            else:
                self._consecutive_tool_errors += 1
                if (
                    self._termination is None
                    and self._consecutive_tool_errors
                    >= self.config.max_consecutive_tool_errors
                ):
                    self._termination = "tool_error_limit"
        if self._termination is None and self._step >= self.config.max_steps:
            self._termination = "max_steps"
        self._step += 1
        return V1EnvironmentStep(
            observation=self._observation(),
            emitted_events=tuple(emitted),
            done=self._termination is not None,
            termination=self._termination,
            tool_result=result,
        )

    def finish(self, agent_name: str) -> V1TrialResult:
        task = self._require_task()
        grade = grade_v1(
            goal=task.goal,
            final_world=self._world,
            events=tuple(self._events),
            audit=tuple(self._audit),
            total_cost=self._total_cost,
            policy_violations=tuple(self._policy_violations),
        )
        return V1TrialResult(
            task_id=task.id,
            agent_name=agent_name,
            termination=self._termination or "incomplete",
            final_world=deepcopy(self._world),
            events=tuple(self._events),
            grade=grade,
            elapsed_ms=(perf_counter() - self._started) * 1000,
        )

    def _tool_step(
        self, task: V1TaskSpec, call: ToolCall
    ) -> tuple[list[TrajectoryEvent], ToolResult]:
        call_event = TrajectoryEvent(
            step=self._step,
            kind="tool_call",
            payload={"name": call.name, "arguments": deepcopy(call.arguments)},
        )
        adapter = adapter_for_name(task.tool_profile, call.name)
        if adapter is None:
            result = ToolResult(False, "unknown_tool", f"unknown tool: {call.name}")
            self._audit.append({"operation": "unknown", "code": result.code})
            return [call_event, self._result_event(call, result)], result
        validation_error = _validate_arguments(adapter, call.arguments)
        if validation_error:
            result = ToolResult(False, "invalid_arguments", validation_error)
            self._audit.append({"operation": adapter.operation, "code": result.code})
            return [call_event, self._result_event(call, result)], result

        self._total_cost += adapter.cost
        policy_events: list[TrajectoryEvent] = []
        if adapter.operation == "search":
            result = self._search(str(call.arguments["query"]))
        elif adapter.operation == "inspect_device":
            result = self._inspect_device(str(call.arguments["device_id"]))
        elif adapter.operation == "inspect_routines":
            result = self._inspect_routines(str(call.arguments["device_id"]))
        elif adapter.operation == "apply_change":
            result, violation = self._apply_change(task, call.arguments)
            if violation:
                self._policy_violations.append(violation)
                policy_events.append(
                    TrajectoryEvent(
                        step=self._step,
                        kind="policy_violation",
                        payload={"code": violation, "tool": call.name},
                    )
                )
        elif adapter.operation == "poll":
            result = self._poll(str(call.arguments["operation_id"]))
        elif adapter.operation == "rollback":
            result = self._rollback(str(call.arguments["operation_id"]))
        else:  # pragma: no cover - catalog and dispatch evolve together
            raise AssertionError(adapter.operation)

        dynamic_events = self._apply_dynamic_events(task, adapter.operation)
        audit = {
            "operation": adapter.operation,
            "code": result.code,
            "ok": result.ok,
            "mutation_attempted": adapter.operation in {"apply_change", "rollback"},
            "effect_applied": bool(result.details.get("effect_applied")),
        }
        if adapter.operation == "apply_change":
            audit.update(
                {
                    "action": call.arguments["action"],
                    "target_id": call.arguments["target_id"],
                }
            )
        self._audit.append(audit)
        return [
            call_event,
            self._result_event(call, result),
            *policy_events,
            *dynamic_events,
        ], result

    def _apply_dynamic_events(
        self, task: V1TaskSpec, operation: str
    ) -> list[TrajectoryEvent]:
        self._operation_invocations[operation] += 1
        emitted = []
        for event in task.dynamic_plan:
            if (
                event.trigger_operation != operation
                or event.invocation != self._operation_invocations[operation]
            ):
                continue
            for path, value in event.changes.items():
                _write_path(self._world, path, deepcopy(value))
            emitted.append(
                TrajectoryEvent(
                    step=self._step,
                    kind="world_event",
                    payload={"message": event.notification},
                )
            )
        return emitted

    def _dialogue_step(
        self, task: V1TaskSpec, action: DialogueAction
    ) -> list[TrajectoryEvent]:
        payload = {
            "kind": action.kind,
            "content": action.content,
            "field": action.field,
            "operation": action.operation,
        }
        emitted = [TrajectoryEvent(step=self._step, kind="message", payload=payload)]
        if action.kind == "clarify":
            if not action.field:
                emitted.append(
                    TrajectoryEvent(
                        step=self._step,
                        kind="interaction_error",
                        payload={"code": "missing_field"},
                    )
                )
            else:
                self._clarified.add(action.field)
                response = task.user_script.get(f"clarify:{action.field}", "unknown")
                emitted.append(
                    TrajectoryEvent(
                        step=self._step,
                        kind="user",
                        payload={"field": action.field, "content": response},
                    )
                )
        elif action.kind == "confirm":
            if not action.operation:
                emitted.append(
                    TrajectoryEvent(
                        step=self._step,
                        kind="interaction_error",
                        payload={"code": "missing_operation"},
                    )
                )
            else:
                response = task.user_script.get(f"confirm:{action.operation}", "no")
                if response == "yes":
                    self._confirmed.add(action.operation)
                emitted.append(
                    TrajectoryEvent(
                        step=self._step,
                        kind="user",
                        payload={"operation": action.operation, "content": response},
                    )
                )
        elif action.kind in {"answer", "refuse"}:
            self._termination = "agent_finished"
        return emitted

    def _search(self, query: str) -> ToolResult:
        normalized = query.strip().lower()
        matches = [
            {
                "device_id": device["id"],
                "label": device["label"],
                "room": device["room"],
            }
            for device in self._world["devices"].values()
            if normalized in " ".join(
                (device["id"], device["label"], device["alias"], device["room"])
            ).lower()
        ]
        return ToolResult(True, "ok", "candidate devices", data={"matches": matches})

    def _inspect_device(self, device_id: str) -> ToolResult:
        device = self._world["devices"].get(device_id)
        if device is None:
            return ToolResult(False, "not_found", "device not found")
        return ToolResult(True, "ok", "device state", data=deepcopy(device))

    def _inspect_routines(self, device_id: str) -> ToolResult:
        if device_id not in self._world["devices"]:
            return ToolResult(False, "not_found", "device not found")
        routines = [
            deepcopy(routine)
            for routine in self._world["routines"].values()
            if routine["device_id"] == device_id
        ]
        return ToolResult(True, "ok", "related routines", data={"routines": routines})

    def _apply_change(
        self, task: V1TaskSpec, arguments: dict[str, Any]
    ) -> tuple[ToolResult, str | None]:
        action = str(arguments["action"])
        target_id = str(arguments["target_id"])
        payload = dict(arguments["payload"])
        key = str(arguments["idempotency_key"])
        if key in self._idempotency:
            operation_id = self._idempotency[key]
            operation = self._world["operations"][operation_id]
            return (
                ToolResult(
                    True,
                    "idempotent_replay",
                    "existing operation returned",
                    data={"operation_id": operation_id, "status": operation["status"]},
                ),
                None,
            )
        if task.goal.required_clarifications and not set(
            task.goal.required_clarifications
        ).issubset(self._clarified):
            return (
                ToolResult(False, "clarification_required", "resolve target ambiguity first"),
                "action_before_clarification",
            )
        target = self._world["devices"].get(target_id) or self._world["routines"].get(
            target_id
        )
        if target is None:
            return ToolResult(False, "not_found", "mutation target not found"), None
        confirmation_key = f"{action}:{target_id}"
        needs_confirmation = action in task.safety_policy.high_impact_actions or (
            task.safety_policy.shared_change_requires_confirmation
            and bool(target.get("shared"))
            and target.get("owner") != "user"
        )
        if needs_confirmation and confirmation_key not in self._confirmed:
            return (
                ToolResult(
                    False,
                    "confirmation_required",
                    "explicit confirmation required for this shared or high-impact change",
                    details={"confirmation_key": confirmation_key},
                ),
                "action_before_confirmation",
            )

        self._invocations[action] += 1
        fault = next(
            (
                item
                for item in task.fault_plan
                if item.action == action and item.invocation == self._invocations[action]
            ),
            None,
        )
        self._operation_index += 1
        operation_id = f"op-{task.id}-{self._operation_index:02d}"
        self._idempotency[key] = operation_id
        self._snapshots[operation_id] = deepcopy(self._world)
        operation = {
            "id": operation_id,
            "action": action,
            "target_id": target_id,
            "payload": deepcopy(payload),
            "status": "pending",
            "idempotency_key": key,
        }
        self._world["operations"][operation_id] = operation

        if fault and fault.kind in {"dropped_write", "async_failure"}:
            operation["status"] = "failed"
            operation["failure"] = fault.kind
            return (
                ToolResult(
                    True,
                    "accepted",
                    "operation accepted",
                    data={"operation_id": operation_id, "status": "pending"},
                ),
                None,
            )

        error = self._mutate(action, target_id, payload)
        if error is not None:
            operation["status"] = "failed"
            operation["failure"] = error.code
            return error, None
        operation["status"] = "completed"
        if fault and fault.kind == "timeout_after_commit":
            return (
                ToolResult(
                    False,
                    "timeout",
                    "request timed out after the operation may have committed",
                    data={"operation_id": operation_id},
                    details={"effect_applied": True},
                ),
                None,
            )
        return (
            ToolResult(
                True,
                "accepted",
                "operation accepted",
                data={"operation_id": operation_id, "status": "pending"},
                details={"effect_applied": True},
            ),
            None,
        )

    def _mutate(self, action: str, target_id: str, payload: dict[str, Any]) -> ToolResult | None:
        if action == "pause_routine":
            routine = self._world["routines"].get(target_id)
            if routine is None:
                return ToolResult(False, "not_found", "routine not found")
            routine["active"] = False
            return None
        device = self._world["devices"].get(target_id)
        if device is None:
            return ToolResult(False, "not_found", "device not found")
        if action != "wake_device" and not device["online"]:
            return ToolResult(False, "precondition_failed", "device is offline")
        if action == "wake_device":
            device["online"] = True
        elif action == "set_mode":
            mode = payload.get("mode")
            if not isinstance(mode, str) or not mode:
                return ToolResult(False, "invalid_arguments", "mode is required")
            device["mode"] = mode
        elif action == "schedule_restore":
            restore_at = payload.get("time")
            if not isinstance(restore_at, str) or not restore_at:
                return ToolResult(False, "invalid_arguments", "restore time is required")
            device["restore_at"] = restore_at
        else:
            return ToolResult(False, "invalid_arguments", f"unsupported action: {action}")
        return None

    def _poll(self, operation_id: str) -> ToolResult:
        operation = self._world["operations"].get(operation_id)
        if operation is None:
            return ToolResult(False, "not_found", "operation not found")
        return ToolResult(
            True,
            "ok",
            "operation status",
            data={
                "operation_id": operation_id,
                "status": operation["status"],
                "failure": operation.get("failure"),
            },
        )

    def _rollback(self, operation_id: str) -> ToolResult:
        operation = self._world["operations"].get(operation_id)
        snapshot = self._snapshots.get(operation_id)
        if operation is None or snapshot is None:
            return ToolResult(False, "not_found", "operation not found")
        if operation["status"] != "completed":
            return ToolResult(False, "not_applicable", "operation did not commit")
        current_operations = deepcopy(self._world["operations"])
        self._world = deepcopy(snapshot)
        self._world["operations"] = current_operations
        self._world["operations"][operation_id]["status"] = "rolled_back"
        return ToolResult(
            True,
            "ok",
            "operation rolled back",
            details={"effect_applied": True},
        )

    def _result_event(self, call: ToolCall, result: ToolResult) -> TrajectoryEvent:
        return TrajectoryEvent(
            step=self._step,
            kind="tool_result",
            payload={
                "name": call.name,
                "ok": result.ok,
                "code": result.code,
                "message": result.message,
                "state_delta": deepcopy(result.state_delta),
                "data": deepcopy(result.data),
                "details": deepcopy(result.details),
            },
        )

    def _observation(self) -> V1Observation:
        task = self._require_task()
        return task.public_observation(step=self._step, events=tuple(self._events))

    def _require_task(self) -> V1TaskSpec:
        if self._task is None:
            raise RuntimeError("reset must be called before using the environment")
        return self._task


def _validate_arguments(adapter: ToolAdapter, arguments: object) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments must be an object"
    schema = adapter.public.parameters
    properties = schema["properties"]
    extras = sorted(set(arguments) - set(properties))
    if extras:
        return f"unexpected arguments: {extras}"
    missing = sorted(set(schema.get("required", ())) - set(arguments))
    if missing:
        return f"missing required arguments: {missing}"
    for key, value in arguments.items():
        expected = properties[key].get("type")
        if expected == "string" and not isinstance(value, str):
            return f"{key} must be a string"
        if expected == "object" and not isinstance(value, dict):
            return f"{key} must be an object"
        if "enum" in properties[key] and value not in properties[key]["enum"]:
            return f"{key} must be one of {properties[key]['enum']}"
    return None


def _write_path(world: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: dict[str, Any] = world
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"dynamic path does not resolve: {path}")
        current = child
    if parts[-1] not in current:
        raise ValueError(f"dynamic path does not resolve: {path}")
    current[parts[-1]] = value
