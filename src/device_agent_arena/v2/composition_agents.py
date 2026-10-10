from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

from device_agent_arena.domain.models import ToolCall, TrajectoryEvent
from device_agent_arena.v1.models import DialogueAction, V1Observation
from device_agent_arena.v2.composition_models import V2CompositionTask


class OracleCompositionAgent:
    name = "oracle-v2-composition-v0.1"

    def __init__(self, task: V2CompositionTask) -> None:
        self._plan = deque(task.oracle_plan)

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction:
        if not self._plan:
            return ToolCall("finish_task", {"message": "任务已完成。"})
        step = self._plan.popleft()
        if step.kind != "tool":
            return DialogueAction(
                kind=step.kind,  # type: ignore[arg-type]
                content=step.content,
                field=step.field,
                operation=step.operation,
            )
        return ToolCall(step.name, _resolve_placeholders(step.arguments, observation.events))


class FinishImmediatelyCompositionAgent:
    name = "finish-immediately-v2-composition-v0.1"

    def act(self, observation: V1Observation) -> ToolCall:
        return ToolCall("finish_task", {"message": "已处理。"})


class RefuseImmediatelyCompositionAgent:
    name = "refuse-immediately-v2-composition-v0.1"

    def act(self, observation: V1Observation) -> ToolCall:
        return ToolCall("refuse_task", {"reason": "无法执行。"})


class PublicReactiveCompositionAgent:
    """Reasonable public-state baseline with no hidden task or oracle access.

    It handles ordinary dependency chains and evidence-backed refusal, but deliberately
    stops after a failed user-side action instead of attempting coordination recovery.
    """

    name = "public-reactive-v2-composition-v0.1"

    def __init__(self, *, recover_user_failure: bool = False) -> None:
        self._target_id: str | None = None
        self._target: dict[str, Any] = {}
        self._relations_seen = False
        self._after_user_action = False
        self._recover_user_failure = recover_user_failure
        self._user_action_key: str | None = None
        self._user_action_failed = False
        self._dependency_operation_id: str | None = None

    def act(self, observation: V1Observation) -> ToolCall:
        if self._target_id is None:
            self._target_id = str(observation.public_context["target_id"])
            return ToolCall("inspect_target", {"target_id": self._target_id})

        recent = _recent_events(observation.events)
        result = next(
            (event for event in reversed(recent) if event.kind == "tool_result"),
            None,
        )
        if result is None:
            return ToolCall("refuse_task", {"reason": "缺少可验证的工具结果。"})
        name = str(result.payload.get("name"))
        data = result.payload.get("data", {})

        if name == "inspect_target" and isinstance(data, dict):
            entity_id = str(data.get("id", ""))
            if entity_id == self._target_id:
                self._target = dict(data)
                if self._after_user_action:
                    if bool(data.get("ready")):
                        return self._apply_target()
                    if (
                        self._recover_user_failure
                        and self._user_action_failed
                        and self._user_action_key is not None
                    ):
                        self._user_action_failed = False
                        return ToolCall(
                            "request_user_action",
                            {
                                "action_key": self._user_action_key,
                                "message": (
                                    "上次用户侧动作未完成，请修正后再次执行公开依赖中"
                                    "列出的动作。"
                                ),
                            },
                        )
                    return ToolCall(
                        "refuse_task",
                        {"reason": "用户侧动作返回后重新观测仍未就绪。"},
                    )
                if self._dependency_operation_id is not None:
                    return self._apply_target()
                if not self._relations_seen:
                    self._relations_seen = True
                    return ToolCall(
                        "inspect_relations", {"target_id": self._target_id}
                    )
                return self._apply_target()
            available = list(data.get("available_actions", []))
            if available:
                return ToolCall(
                    "apply_action",
                    {"action": str(available[0]), "target_id": entity_id, "payload": {}},
                )

        if name == "inspect_relations" and isinstance(data, dict):
            relations = list(data.get("relations", []))
            related = dict(data.get("related_entities", {}))
            if not bool(self._target.get("authorized", True)):
                return ToolCall(
                    "refuse_task", {"reason": "公开状态显示当前缺少目标授权。"}
                )
            user_relation = next(
                (item for item in relations if item.get("user_action_key")), None
            )
            if user_relation is not None:
                if not bool(user_relation.get("user_action_available")):
                    return ToolCall(
                        "refuse_task",
                        {"reason": "所需用户侧动作当前不可用，无法安全执行。"},
                    )
                self._after_user_action = True
                self._user_action_key = str(user_relation["user_action_key"])
                return ToolCall(
                    "request_user_action",
                    {
                        "action_key": self._user_action_key,
                        "message": "请完成公开依赖中列出的用户侧动作。",
                    },
                )
            if any(item.get("relation") == "satisfied_dependency" for item in relations):
                return self._apply_target()
            dependency = next(
                (
                    entity
                    for entity in related.values()
                    if entity.get("available_actions")
                ),
                None,
            )
            if dependency is not None:
                return ToolCall(
                    "inspect_target", {"target_id": str(dependency["id"])}
                )
            return self._apply_target()

        if name == "request_user_action":
            self._user_action_failed = (
                isinstance(data, dict) and str(data.get("response")) == "failed"
            )
            return ToolCall("inspect_target", {"target_id": self._target_id})

        if name == "apply_action" and isinstance(data, dict):
            operation_id = data.get("operation_id")
            if not operation_id:
                return ToolCall("refuse_task", {"reason": "动作未返回 operation ID。"})
            return ToolCall("poll_action", {"operation_id": str(operation_id)})

        if name == "poll_action" and isinstance(data, dict):
            if str(data.get("target_id")) == self._target_id:
                return ToolCall("finish_task", {"message": "已验证目标动作完成。"})
            self._dependency_operation_id = str(data["id"])
            return ToolCall("inspect_target", {"target_id": self._target_id})

        return ToolCall("refuse_task", {"reason": "无法从公开状态继续安全执行。"})

    def _apply_target(self) -> ToolCall:
        available = list(self._target.get("available_actions", []))
        if not available or self._target_id is None:
            return ToolCall("refuse_task", {"reason": "目标没有公开可执行动作。"})
        payload = (
            {"dependency_operation_id": self._dependency_operation_id}
            if self._dependency_operation_id is not None
            else {}
        )
        return ToolCall(
            "apply_action",
            {
                "action": str(available[0]),
                "target_id": self._target_id,
                "payload": payload,
            },
        )


class PublicRecoveryCompositionAgent(PublicReactiveCompositionAgent):
    """Public-contract controller with user-side failure reobservation and retry."""

    name = "public-recovery-v2-composition-v0.1"

    def __init__(self) -> None:
        super().__init__(recover_user_failure=True)


def _resolve_placeholders(value: Any, events: tuple[TrajectoryEvent, ...]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_placeholders(item, events) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(item, events) for item in value]
    if value != "$last_operation":
        return deepcopy(value)
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
    return operation_id


def _recent_events(events: tuple[TrajectoryEvent, ...]) -> tuple[TrajectoryEvent, ...]:
    if not events:
        return ()
    step = max(event.step for event in events)
    return tuple(event for event in events if event.step == step)
