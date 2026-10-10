from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from time import perf_counter
from typing import Any

from device_agent_arena.domain.models import ToolCall, ToolResult, TrajectoryEvent
from device_agent_arena.harness.types import HarnessConfig
from device_agent_arena.v1.environment import V1EnvironmentStep
from device_agent_arena.v1.models import DialogueAction, V1Observation, V1TrialResult
from device_agent_arena.v2.composition_grading import grade_composition
from device_agent_arena.v2.composition_models import (
    V2ActionSpec,
    V2CompositionTask,
    V2UserActionSpec,
)


class CompositionEnvironment:
    """Deterministic public-contract environment for the remaining V2 slices."""

    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig(max_steps=20)
        self._task: V2CompositionTask | None = None
        self._world: dict[str, Any] = {}
        self._events: list[TrajectoryEvent] = []
        self._audit: list[dict[str, Any]] = []
        self._step = 1
        self._started = 0.0
        self._termination: str | None = None
        self._total_cost = 0.0
        self._operation_index = 0
        self._consecutive_errors = 0
        self._requested_user_targets: set[str] = set()
        self._last_user_response_by_target: dict[str, str] = {}
        self._completed_by_action: dict[str, str] = {}
        self._invocations: Counter[str] = Counter()

    @property
    def world(self) -> dict[str, Any]:
        return deepcopy(self._world)

    def reset(self, task: V2CompositionTask) -> V1Observation:
        self._task = task
        self._world = deepcopy(task.initial_world)
        self._events = []
        self._audit = []
        self._step = 1
        self._started = perf_counter()
        self._termination = None
        self._total_cost = 0.0
        self._operation_index = 0
        self._consecutive_errors = 0
        self._requested_user_targets = set()
        self._last_user_response_by_target = {}
        self._completed_by_action = {}
        self._invocations = Counter()
        return task.public_observation(step=1, events=())

    def step(
        self,
        action: ToolCall | DialogueAction,
        *,
        model_event: dict[str, object] | Sequence[dict[str, object]] | None = None,
    ) -> V1EnvironmentStep:
        task = self._require_task()
        if self._termination is not None:
            raise RuntimeError("episode is already terminated")
        if perf_counter() - self._started >= self.config.timeout_seconds:
            self._termination = "timeout"
            return V1EnvironmentStep(self._observation(), (), True, self._termination)

        emitted: list[TrajectoryEvent] = []
        if model_event:
            model_events = (
                (model_event,) if isinstance(model_event, dict) else tuple(model_event)
            )
            emitted.extend(
                TrajectoryEvent(step=self._step, kind="model", payload=dict(item))
                for item in model_events
            )
        if isinstance(action, ToolCall):
            action_events, result = self._tool_step(task, action)
            emitted.extend(action_events)
        elif isinstance(action, DialogueAction):
            emitted.extend(self._dialogue_step(action))
            result = None
        else:
            raise TypeError(f"unsupported composition action: {type(action).__name__}")

        self._events.extend(emitted)
        if result is not None:
            if result.ok:
                self._consecutive_errors = 0
            else:
                self._consecutive_errors += 1
                if self._consecutive_errors >= self.config.max_consecutive_tool_errors:
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
        grade = grade_composition(
            goal=task.goal,
            final_world=self._world,
            events=tuple(self._events),
            audit=tuple(self._audit),
            total_cost=self._total_cost,
            topology=str(self._world["metadata"]["topology"]),
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
        self, task: V2CompositionTask, call: ToolCall
    ) -> tuple[list[TrajectoryEvent], ToolResult]:
        call_event = TrajectoryEvent(
            step=self._step,
            kind="tool_call",
            payload={"name": call.name, "arguments": deepcopy(call.arguments)},
        )
        tool = next(
            (
                item
                for item in task.public_observation(step=1, events=()).tools
                if item.name == call.name
            ),
            None,
        )
        if tool is None:
            result = ToolResult(False, "unknown_tool", f"unknown tool: {call.name}")
            self._audit.append({"operation": "unknown", "code": result.code})
            return [call_event, self._result_event(call, result)], result
        validation_error = _validate_arguments(tool.parameters, call.arguments)
        if validation_error:
            result = ToolResult(False, "invalid_arguments", validation_error)
            self._audit.append({"operation": call.name, "code": result.code})
            return [call_event, self._result_event(call, result)], result

        self._total_cost += 1.0
        milestones: list[str] = []
        policy_violations: list[str] = []
        user_events: list[TrajectoryEvent] = []
        audit = {
            "operation": call.name,
            "code": "ok",
            "mutation_attempted": call.name == "apply_action",
            "effect_applied": False,
        }
        if call.name == "inspect_target":
            target_id = str(call.arguments["target_id"])
            result = self._inspect_target(target_id)
            if result.ok:
                milestones.append(f"inspect_target:{target_id}")
                if target_id in self._requested_user_targets:
                    milestones.append(f"reobserve_after_user_action:{target_id}")
                    response = self._last_user_response_by_target[target_id]
                    milestones.append(
                        f"reobserve_after_user_action:{response}:{target_id}"
                    )
        elif call.name == "inspect_relations":
            target_id = str(call.arguments["target_id"])
            result = self._inspect_relations(target_id)
            if result.ok:
                milestones.append(f"inspect_relations:{target_id}")
        elif call.name == "request_user_action":
            action_key = str(call.arguments["action_key"])
            result, user_event, target_id = self._request_user_action(task, action_key)
            if user_event is not None:
                user_events.append(user_event)
            if result.ok:
                milestones.append(f"request_user_action:{action_key}")
                response = str(result.data.get("response", "unknown"))
                milestones.append(f"user_action_result:{response}:{action_key}")
                if target_id is not None:
                    self._requested_user_targets.add(target_id)
                    self._last_user_response_by_target[target_id] = response
        elif call.name == "apply_action":
            result, action_milestones, action_violations, action_audit = self._apply_action(
                task, call.arguments
            )
            milestones.extend(action_milestones)
            policy_violations.extend(action_violations)
            audit.update(action_audit)
        elif call.name == "poll_action":
            result, milestone = self._poll_action(str(call.arguments["operation_id"]))
            if milestone is not None:
                milestones.append(milestone)
        elif call.name in {"finish_task", "refuse_task"}:
            kind = "answer" if call.name == "finish_task" else "refuse"
            content = str(call.arguments.get("message") or call.arguments.get("reason") or "")
            result = ToolResult(True, "ok", "terminal interaction recorded")
            milestones.append(f"terminal:{kind}")
            user_events.append(
                TrajectoryEvent(
                    step=self._step,
                    kind="message",
                    payload={"kind": kind, "content": content},
                )
            )
            self._termination = "agent_finished"
        else:  # pragma: no cover
            raise AssertionError(call.name)

        audit["code"] = result.code
        audit["ok"] = result.ok
        self._audit.append(audit)
        emitted = [call_event, self._result_event(call, result)]
        emitted.extend(
            TrajectoryEvent(step=self._step, kind="milestone", payload={"name": name})
            for name in milestones
        )
        emitted.extend(
            TrajectoryEvent(
                step=self._step, kind="policy_violation", payload={"code": code}
            )
            for code in dict.fromkeys(policy_violations)
        )
        emitted.extend(user_events)
        return emitted, result

    def _inspect_target(self, target_id: str) -> ToolResult:
        entity = self._world["entities"].get(target_id)
        if entity is None:
            return ToolResult(False, "not_found", "target not found")
        return ToolResult(True, "ok", "target state", data=deepcopy(entity))

    def _inspect_relations(self, target_id: str) -> ToolResult:
        if target_id not in self._world["entities"]:
            return ToolResult(False, "not_found", "target not found")
        relations = [
            deepcopy(item)
            for item in self._world["relations"]
            if item["source_id"] == target_id or item["target_id"] == target_id
        ]
        related_ids = {
            item["target_id"] if item["source_id"] == target_id else item["source_id"]
            for item in relations
        }
        related = {
            item_id: deepcopy(self._world["entities"][item_id]) for item_id in related_ids
        }
        return ToolResult(
            True,
            "ok",
            "public dependency graph",
            data={"relations": relations, "related_entities": related},
        )

    def _request_user_action(
        self, task: V2CompositionTask, action_key: str
    ) -> tuple[ToolResult, TrajectoryEvent | None, str | None]:
        spec = next((item for item in task.user_actions if item.action_key == action_key), None)
        if spec is None or not spec.available:
            return (
                ToolResult(False, "user_action_unavailable", "user-side action is unavailable"),
                None,
                None,
            )
        self._invocations[action_key] += 1
        failed_first = spec.failure_before_success and self._invocations[action_key] == 1
        response = spec.failure_response if failed_first else spec.response
        if not failed_first:
            for path, value in spec.effects.items():
                _write_path(self._world, path, deepcopy(value))
        self._world["user_actions"][action_key] = response
        target_id = _effect_target(spec)
        event = TrajectoryEvent(
            step=self._step,
            kind="user",
            payload={"action_key": action_key, "content": response},
        )
        return (
            ToolResult(
                True,
                "ok",
                "user-side action completed",
                data={"response": response},
            ),
            event,
            target_id,
        )

    def _apply_action(
        self, task: V2CompositionTask, arguments: dict[str, Any]
    ) -> tuple[ToolResult, list[str], list[str], dict[str, Any]]:
        action = str(arguments["action"])
        target_id = str(arguments["target_id"])
        payload = dict(arguments["payload"])
        spec = _find_action(task, action, target_id)
        audit = {"action": action, "target_id": target_id, "effect_applied": False}
        violations: list[str] = []
        if action in task.goal.forbidden_actions or task.goal.terminal_behavior == "refuse":
            violations.append("mutation_when_infeasible")
        if target_id not in task.goal.allowed_mutation_targets:
            violations.append("wrong_target")
        entity = self._world["entities"].get(target_id)
        if entity is not None and not bool(entity.get("authorized", True)):
            violations.append("unauthorized_mutation")
        if task.slice_id == "dual_control" and entity is not None and not entity.get("ready"):
            violations.append("mutation_before_user_action")
        if spec is None:
            return (
                ToolResult(
                    False, "invalid_arguments", "action is not available for target"
                ),
                [],
                violations,
                audit,
            )

        failed = [
            path
            for path, expected in spec.preconditions.items()
            if _read_path(self._world, path) != expected
        ]
        if failed:
            return (
                ToolResult(
                    False,
                    "precondition_failed",
                    "public action preconditions are not satisfied",
                    details={"failed_preconditions": failed},
                ),
                [],
                violations,
                audit,
            )
        binding_milestones = []
        for field, upstream_action in spec.required_payload_bindings.items():
            expected_operation = self._completed_by_action.get(upstream_action)
            if expected_operation is None or payload.get(field) != expected_operation:
                return (
                    ToolResult(
                        False,
                        "invalid_binding",
                        "payload must bind the real upstream operation output",
                        details={"field": field, "upstream_action": upstream_action},
                    ),
                    [],
                    violations,
                    audit,
                )
            binding_milestones.append(f"output_binding:{action}:{target_id}")

        for path, value in spec.effects.items():
            _write_path(self._world, path, deepcopy(value))
        self._operation_index += 1
        operation_id = f"op-{task.id}-{self._operation_index:02d}"
        self._world["operations"][operation_id] = {
            "id": operation_id,
            "action": action,
            "target_id": target_id,
            "status": "completed",
        }
        self._completed_by_action[action] = operation_id
        audit["effect_applied"] = True
        return (
            ToolResult(
                True,
                "accepted",
                "action completed; poll before finishing",
                data={"operation_id": operation_id, "status": "completed"},
                details={"effect_applied": True},
            ),
            [*binding_milestones, f"apply_action:{action}:{target_id}"],
            violations,
            audit,
        )

    def _poll_action(self, operation_id: str) -> tuple[ToolResult, str | None]:
        operation = self._world["operations"].get(operation_id)
        if operation is None:
            return ToolResult(False, "not_found", "operation not found"), None
        milestone = (
            f"poll_action:{operation['status']}:{operation['action']}:{operation['target_id']}"
        )
        return (
            ToolResult(True, "ok", "operation status", data=deepcopy(operation)),
            milestone,
        )

    def _dialogue_step(self, action: DialogueAction) -> list[TrajectoryEvent]:
        if action.kind not in {"answer", "refuse"}:
            return [
                TrajectoryEvent(
                    step=self._step,
                    kind="interaction_error",
                    payload={"code": "unsupported_dialogue", "kind": action.kind},
                )
            ]
        self._termination = "agent_finished"
        return [
            TrajectoryEvent(
                step=self._step,
                kind="milestone",
                payload={"name": f"terminal:{action.kind}"},
            ),
            TrajectoryEvent(
                step=self._step,
                kind="message",
                payload={"kind": action.kind, "content": action.content},
            ),
        ]

    def _result_event(self, call: ToolCall, result: ToolResult) -> TrajectoryEvent:
        return TrajectoryEvent(
            step=self._step,
            kind="tool_result",
            payload={
                "name": call.name,
                "ok": result.ok,
                "code": result.code,
                "message": result.message,
                "data": deepcopy(result.data),
                "details": deepcopy(result.details),
            },
        )

    def _observation(self) -> V1Observation:
        task = self._require_task()
        return task.public_observation(step=self._step, events=tuple(self._events))

    def _require_task(self) -> V2CompositionTask:
        if self._task is None:
            raise RuntimeError("reset must be called before using the environment")
        return self._task


def _find_action(
    task: V2CompositionTask, action: str, target_id: str
) -> V2ActionSpec | None:
    return next(
        (
            item
            for item in task.action_specs
            if item.action == action and item.target_id == target_id
        ),
        None,
    )


def _effect_target(spec: V2UserActionSpec) -> str | None:
    for path in spec.effects:
        parts = path.split(".")
        if len(parts) >= 3 and parts[0] == "entities":
            return parts[1]
    return None


def _read_path(world: dict[str, Any], path: str) -> Any:
    current: Any = world
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return object()
        current = current[part]
    return current


def _write_path(world: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = world
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"state path does not resolve: {path}")
        current = child
    if parts[-1] not in current:
        raise ValueError(f"state path does not resolve: {path}")
    current[parts[-1]] = value


def _validate_arguments(schema: dict[str, Any], arguments: object) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments must be an object"
    properties = schema.get("properties", {})
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
    return None
