from __future__ import annotations

from typing import Any, Protocol

from device_agent_arena.domain.models import ToolCall, TrajectoryEvent
from device_agent_arena.v1.models import DialogueAction, V1Observation


class TelemetryAgent(Protocol):
    name: str

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction: ...

    def drain_telemetry(self) -> dict[str, Any] | None: ...


def _recent_events(events: tuple[TrajectoryEvent, ...]) -> tuple[TrajectoryEvent, ...]:
    if not events:
        return ()
    step = max(event.step for event in events)
    return tuple(event for event in events if event.step == step)


def expected_fast_path_action(observation: V1Observation) -> ToolCall | None:
    """Return the only action a speculative router may take in a safe fast state."""
    target_id = str(observation.public_context["target_id"])
    recent = _recent_events(observation.events)
    if not recent:
        return ToolCall("inspect_target", {"target_id": target_id})
    result = next(
        (event for event in reversed(recent) if event.kind == "tool_result"),
        None,
    )
    if result is None:
        return None
    payload = result.payload
    name = str(payload.get("name", ""))
    data = payload.get("data", {})
    if (
        name == "request_user_action"
        and payload.get("ok")
        and isinstance(data, dict)
        and str(data.get("response")) in {"done", "failed"}
    ):
        return ToolCall("inspect_target", {"target_id": target_id})
    if name == "apply_action" and payload.get("ok") and isinstance(data, dict):
        operation_id = data.get("operation_id")
        if operation_id:
            return ToolCall("poll_action", {"operation_id": str(operation_id)})
    if (
        name == "poll_action"
        and payload.get("ok")
        and isinstance(data, dict)
        and str(data.get("status")) == "completed"
        and str(data.get("target_id")) == target_id
    ):
        return ToolCall("finish_task", {"message": "已验证目标动作完成。"})
    return None


def _matches_fast_path(
    proposed: ToolCall | DialogueAction,
    expected: ToolCall,
) -> bool:
    if not isinstance(proposed, ToolCall) or proposed.name != expected.name:
        return False
    if expected.name == "finish_task":
        return bool(str(proposed.arguments.get("message", "")).strip())
    return proposed.arguments == expected.arguments


class LayeredCompositionAgent:
    """Speculative small-model fast path with exact fallback to the main planner."""

    def __init__(self, *, router: TelemetryAgent, planner: TelemetryAgent) -> None:
        self.router = router
        self.planner = planner
        self._telemetry: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return f"layered:{self.router.name}+{self.planner.name}"

    def _record(
        self,
        agent: TelemetryAgent,
        *,
        role: str,
        decision_used: bool,
        rejection_reason: str | None = None,
    ) -> None:
        telemetry = dict(agent.drain_telemetry() or {})
        telemetry["model_role"] = role
        telemetry["decision_used"] = decision_used
        if rejection_reason is not None:
            telemetry["rejection_reason"] = rejection_reason
        self._telemetry.append(telemetry)

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction:
        self._telemetry = []
        expected = expected_fast_path_action(observation)
        if expected is not None:
            proposed = self.router.act(observation)
            if _matches_fast_path(proposed, expected):
                self._record(self.router, role="router", decision_used=True)
                return proposed
            self._record(
                self.router,
                role="router",
                decision_used=False,
                rejection_reason="fast_path_contract_mismatch",
            )
        action = self.planner.act(observation)
        self._record(self.planner, role="planner", decision_used=True)
        return action

    def drain_telemetry(self) -> tuple[dict[str, Any], ...]:
        telemetry = tuple(self._telemetry)
        self._telemetry = []
        return telemetry
