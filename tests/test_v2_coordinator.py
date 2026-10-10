from __future__ import annotations

from collections import deque

from device_agent_arena.domain.models import ToolCall, TrajectoryEvent
from device_agent_arena.v1.generate import generate_tasks
from device_agent_arena.v2.coordinator import MutationCoordinatorV2


class QueueAgent:
    name = "queue"

    def __init__(self, actions: list[ToolCall]) -> None:
        self.actions = deque(actions)

    def act(self, _observation):
        return self.actions.popleft()


def _mutation() -> ToolCall:
    return ToolCall(
        "apply_change",
        {"action": "set_mode", "target_id": "node-1", "payload": {"mode": "focus"}},
    )


def _observation(events: tuple[TrajectoryEvent, ...] = ()):
    task = generate_tasks("dev", count=2, root_seed=7)[1]
    return task.public_observation(
        step=max((event.step for event in events), default=0), events=events
    )


def test_coordinator_rotates_key_after_definitive_failure() -> None:
    coordinator = MutationCoordinatorV2(QueueAgent([_mutation(), _mutation()]))
    first = coordinator.act(_observation())
    events = (
        TrajectoryEvent(
            step=1,
            kind="tool_call",
            payload={"name": first.name, "arguments": first.arguments},
        ),
        TrajectoryEvent(
            step=1,
            kind="tool_result",
            payload={"code": "precondition_failed", "data": {}, "ok": False},
        ),
    )

    second = coordinator.act(_observation(events))

    assert first.arguments["idempotency_key"] != second.arguments["idempotency_key"]


def test_coordinator_reuses_key_while_commit_status_is_unknown() -> None:
    coordinator = MutationCoordinatorV2(QueueAgent([_mutation(), _mutation()]))
    first = coordinator.act(_observation())
    events = (
        TrajectoryEvent(
            step=1,
            kind="tool_call",
            payload={"name": first.name, "arguments": first.arguments},
        ),
        TrajectoryEvent(
            step=1,
            kind="tool_result",
            payload={
                "code": "timeout",
                "data": {"operation_id": "op-1"},
                "ok": False,
            },
        ),
    )

    second = coordinator.act(_observation(events))

    assert first.arguments["idempotency_key"] == second.arguments["idempotency_key"]


def test_coordinator_rotates_after_poll_reports_failed() -> None:
    coordinator = MutationCoordinatorV2(QueueAgent([_mutation(), _mutation()]))
    first = coordinator.act(_observation())
    events = (
        TrajectoryEvent(
            step=1,
            kind="tool_call",
            payload={"name": first.name, "arguments": first.arguments},
        ),
        TrajectoryEvent(
            step=1,
            kind="tool_result",
            payload={
                "code": "accepted",
                "data": {"operation_id": "op-1", "status": "pending"},
                "ok": True,
            },
        ),
        TrajectoryEvent(
            step=2,
            kind="tool_call",
            payload={"name": "check_job", "arguments": {"operation_id": "op-1"}},
        ),
        TrajectoryEvent(
            step=2,
            kind="tool_result",
            payload={
                "code": "ok",
                "data": {"operation_id": "op-1", "status": "failed"},
                "ok": True,
            },
        ),
    )

    second = coordinator.act(_observation(events))

    assert first.arguments["idempotency_key"] != second.arguments["idempotency_key"]
