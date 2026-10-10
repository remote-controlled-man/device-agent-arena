from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from device_agent_arena.domain.models import ToolCall, TrajectoryEvent
from device_agent_arena.v1.models import DialogueAction, V1Observation
from device_agent_arena.v1.runner import AgentV1


@dataclass
class _Attempt:
    key: str
    status: str = "submitted"


class MutationCoordinatorV2:
    """Own runtime idempotency keys using public calls and results only."""

    def __init__(self, delegate: AgentV1) -> None:
        self.delegate = delegate
        self._processed_events = 0
        self._key_index = 0
        self._attempts: dict[str, _Attempt] = {}
        self._operation_intents: dict[str, str] = {}
        self._telemetry: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return f"mutation-coordinator:{self.delegate.name}"

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction:
        self._ingest(observation.events)
        action = self.delegate.act(observation)
        drain = getattr(self.delegate, "drain_telemetry", None)
        delegate_telemetry = drain() if callable(drain) else None
        self._telemetry = dict(delegate_telemetry or {})
        if not isinstance(action, ToolCall) or not _is_mutation(action, observation):
            return action
        arguments = deepcopy(action.arguments)
        arguments.pop("idempotency_key", None)
        intent = _intent(arguments)
        attempt = self._attempts.get(intent)
        rotated = attempt is not None and attempt.status == "failed"
        if attempt is None or rotated:
            self._key_index += 1
            attempt = _Attempt(key=f"v2-{observation.episode_id}-{self._key_index}")
            self._attempts[intent] = attempt
        arguments["idempotency_key"] = attempt.key
        self._telemetry["mutation_coordinator"] = {
            "intent": intent,
            "key_action": "rotated" if rotated else "allocated_or_reused",
            "attempt_status_before_call": attempt.status,
        }
        return ToolCall(action.name, arguments)

    def drain_telemetry(self) -> dict[str, Any] | None:
        telemetry = self._telemetry
        self._telemetry = None
        return telemetry

    def _ingest(self, events: tuple[TrajectoryEvent, ...]) -> None:
        new_events = events[self._processed_events :]
        self._processed_events = len(events)
        by_step: dict[int, list[TrajectoryEvent]] = {}
        for event in new_events:
            by_step.setdefault(event.step, []).append(event)
        for step_events in by_step.values():
            call = next((event for event in step_events if event.kind == "tool_call"), None)
            result = next((event for event in step_events if event.kind == "tool_result"), None)
            if call is None or result is None:
                continue
            arguments = call.payload.get("arguments", {})
            if not isinstance(arguments, dict):
                continue
            intent: str | None = None
            if {"action", "target_id", "payload"} <= set(arguments):
                intent = _intent(arguments)
            else:
                operation_id = arguments.get("operation_id")
                if isinstance(operation_id, str):
                    intent = self._operation_intents.get(operation_id)
            if intent is None:
                continue
            attempt = self._attempts.get(intent)
            if attempt is None:
                key = str(arguments.get("idempotency_key", "observed"))
                attempt = _Attempt(key=key)
                self._attempts[intent] = attempt
            payload = result.payload
            data = payload.get("data", {})
            if not isinstance(data, dict):
                data = {}
            operation_id = data.get("operation_id")
            if isinstance(operation_id, str):
                self._operation_intents[operation_id] = intent
            code = str(payload.get("code", ""))
            status = str(data.get("status", ""))
            if status == "failed" or code in {
                "precondition_failed",
                "invalid_arguments",
                "permission_denied",
                "not_found",
            }:
                attempt.status = "failed"
            elif status in {"completed", "rolled_back"}:
                attempt.status = status
            elif code == "timeout":
                attempt.status = "unknown"
            elif status == "pending" or code == "accepted":
                attempt.status = "pending"


def _intent(arguments: dict[str, Any]) -> str:
    payload = arguments.get("payload", {})
    return "|".join(
        (
            str(arguments.get("action", "")),
            str(arguments.get("target_id", "")),
            repr(sorted(payload.items())) if isinstance(payload, dict) else repr(payload),
        )
    )


def _is_mutation(action: ToolCall, observation: V1Observation) -> bool:
    tool = next((item for item in observation.tools if item.name == action.name), None)
    if tool is None:
        return False
    properties = tool.parameters.get("properties", {})
    return {"action", "target_id", "payload"} <= set(properties)
