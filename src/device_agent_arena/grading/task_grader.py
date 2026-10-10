from __future__ import annotations

from collections.abc import Iterable

from device_agent_arena.domain.models import (
    DeviceState,
    GoalSpec,
    GradeResult,
    TrajectoryEvent,
)

MUTATING_TOOLS = {
    "connect_device",
    "calibrate_device",
    "set_orientation",
    "set_mode",
    "set_tracking",
    "save_preset",
    "recall_preset",
}


def grade_task(
    *, goal: GoalSpec, final_state: DeviceState, events: Iterable[TrajectoryEvent]
) -> GradeResult:
    trajectory = tuple(events)
    checks: dict[str, bool] = {}

    for field, expected in goal.final_state.items():
        actual = getattr(final_state, field, object())
        checks[f"final_state.{field}"] = actual == expected

    called_tools = tuple(
        event.payload.get("name") for event in trajectory if event.kind == "tool_call"
    )
    successful_tools = tuple(
        event.payload.get("name")
        for event in trajectory
        if event.kind == "tool_result" and event.payload.get("ok") is True
    )
    for tool in goal.required_tools:
        checks[f"required_tool.{tool}"] = tool in successful_tools
    for tool in goal.forbidden_tools:
        checks[f"forbidden_tool.{tool}"] = tool not in called_tools

    invalid_tool_results = tuple(
        event.payload.get("code")
        for event in trajectory
        if event.kind == "tool_result"
        and event.payload.get("code") in {"invalid_arguments", "unknown_tool"}
    )
    checks["trajectory.valid_tool_calls"] = not invalid_tool_results

    first_mutation_index = next(
        (
            index
            for index, event in enumerate(trajectory)
            if event.kind == "tool_call" and event.payload.get("name") in MUTATING_TOOLS
        ),
        len(trajectory),
    )
    for field in goal.required_clarifications:
        clarification_index = next(
            (
                index
                for index, event in enumerate(trajectory)
                if event.kind == "message"
                and event.payload.get("kind") == "clarify"
                and event.payload.get("field") == field
            ),
            None,
        )
        checks[f"required_clarification.{field}"] = (
            clarification_index is not None and clarification_index < first_mutation_index
        )

    terminal_kinds = tuple(
        event.payload.get("kind") for event in trajectory if event.kind == "message"
    )
    checks[f"terminal_behavior.{goal.terminal_behavior}"] = (
        bool(terminal_kinds) and terminal_kinds[-1] == goal.terminal_behavior
    )

    return GradeResult(
        success=all(checks.values()),
        checks=checks,
        diagnostics={
            "called_tools": called_tools,
            "successful_tools": successful_tools,
            "invalid_tool_results": invalid_tool_results,
            "terminal_kinds": terminal_kinds,
        },
    )
