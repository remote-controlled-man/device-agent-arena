from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

from device_agent_arena.domain.models import (
    MessageAction,
    TaskSpec,
    ToolCall,
    ToolCallBatch,
    ToolResult,
    TrajectoryEvent,
    TrialResult,
)
from device_agent_arena.grading.task_grader import grade_task
from device_agent_arena.harness.types import AgentContext, HarnessConfig
from device_agent_arena.simulator.device import DeviceSimulator
from device_agent_arena.simulator.tools import INTERACTION_TOOL_NAMES, validate_tool_arguments


@dataclass(frozen=True)
class EnvironmentStep:
    observation: AgentContext
    emitted_events: tuple[TrajectoryEvent, ...]
    done: bool
    termination: str | None = None
    tool_result: ToolResult | None = None
    tool_results: tuple[ToolResult, ...] = ()


class AgentEnvironment:
    """Own one isolated, bounded device-control episode."""

    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig()
        self._task: TaskSpec | None = None
        self._simulator: DeviceSimulator | None = None
        self._events: list[TrajectoryEvent] = []
        self._step = 1
        self._consecutive_errors = 0
        self._termination: str | None = None
        self._started = 0.0

    def reset(self, task: TaskSpec) -> AgentContext:
        self._task = task
        self._simulator = DeviceSimulator(
            deepcopy(task.initial_state),
            task.fault_plan,
            seed=task.seed,
        )
        self._events = []
        self._step = 1
        self._consecutive_errors = 0
        self._termination = None
        self._started = perf_counter()
        return self._observation()

    def step(
        self,
        action: ToolCall | ToolCallBatch | MessageAction,
        *,
        model_event: dict[str, object] | None = None,
    ) -> EnvironmentStep:
        task, simulator = self._require_episode()
        if self._termination is not None:
            raise RuntimeError("episode is already terminated")
        emitted: list[TrajectoryEvent] = []
        if model_event:
            emitted.append(
                TrajectoryEvent(step=self._step, kind="model", payload=dict(model_event))
            )
        if perf_counter() - self._started >= self.config.timeout_seconds:
            self._events.extend(emitted)
            self._termination = "timeout"
            return EnvironmentStep(self._observation(), tuple(emitted), True, self._termination)

        tool_results: list[ToolResult] = []
        if isinstance(action, (ToolCall, ToolCallBatch)):
            calls = (action,) if isinstance(action, ToolCall) else action.calls
            batch_id = f"turn_{self._step}"
            for call in calls:
                if perf_counter() - self._started >= self.config.timeout_seconds:
                    self._termination = "timeout"
                    break
                call_id = f"call_{self._step}"
                emitted.append(
                    TrajectoryEvent(
                        step=self._step,
                        kind="tool_call",
                        payload={
                            "id": call_id,
                            "batch_id": batch_id,
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    )
                )
                interaction_events: list[TrajectoryEvent] = []
                if call.name in INTERACTION_TOOL_NAMES:
                    validation_error = validate_tool_arguments(call.name, call.arguments)
                    if validation_error is not None:
                        tool_result = ToolResult(
                            False,
                            "invalid_arguments",
                            validation_error,
                        )
                    elif call.name == "request_clarification":
                        field = str(call.arguments["field"])
                        content = str(call.arguments["content"])
                        user_content = task.clarifications.get(field)
                        interaction_events.append(
                            TrajectoryEvent(
                                step=self._step,
                                kind="message",
                                payload={
                                    "kind": "clarify",
                                    "content": content,
                                    "field": field,
                                    "via_tool": True,
                                },
                            )
                        )
                        if user_content is not None:
                            interaction_events.append(
                                TrajectoryEvent(
                                    step=self._step,
                                    kind="user",
                                    payload={
                                        "field": field,
                                        "content": user_content,
                                        "via_tool": True,
                                    },
                                )
                            )
                        tool_result = ToolResult(
                            True,
                            "ok",
                            "clarification received",
                            data={"field": field, "content": user_content},
                        )
                    else:
                        kind = str(call.arguments["kind"])
                        content = str(call.arguments["content"])
                        tool_result = ToolResult(True, "ok", "task finished")
                        interaction_events.append(
                            TrajectoryEvent(
                                step=self._step,
                                kind="message",
                                payload={
                                    "kind": kind,
                                    "content": content,
                                    "field": None,
                                    "via_tool": True,
                                },
                            )
                        )
                        self._termination = "agent_finished"
                else:
                    tool_result = simulator.execute(call)
                tool_results.append(tool_result)
                emitted.append(
                    TrajectoryEvent(
                        step=self._step,
                        kind="tool_result",
                        payload={
                            "id": call_id,
                            "batch_id": batch_id,
                            "name": call.name,
                            "ok": tool_result.ok,
                            "code": tool_result.code,
                            "message": tool_result.message,
                            "state_delta": tool_result.state_delta,
                            "data": tool_result.data,
                            "details": tool_result.details,
                        },
                    )
                )
                emitted.extend(interaction_events)
                self._consecutive_errors = (
                    0 if tool_result.ok else self._consecutive_errors + 1
                )
                if self._consecutive_errors >= self.config.max_consecutive_tool_errors:
                    self._termination = "tool_error_limit"
                if self._termination is None and self._step >= self.config.max_steps:
                    self._termination = "max_steps"
                self._step += 1
                if self._termination is not None:
                    break
        elif isinstance(action, MessageAction):
            emitted.append(
                TrajectoryEvent(
                    step=self._step,
                    kind="message",
                    payload={"kind": action.kind, "content": action.content, "field": action.field},
                )
            )
            if action.kind in {"answer", "refuse"}:
                self._termination = "agent_finished"
            elif action.kind == "clarify" and action.field in task.clarifications:
                emitted.append(
                    TrajectoryEvent(
                        step=self._step,
                        kind="user",
                        payload={
                            "field": action.field,
                            "content": task.clarifications[action.field],
                        },
                    )
                )
            if self._termination is None and self._step >= self.config.max_steps:
                self._termination = "max_steps"
            self._step += 1
        else:
            raise TypeError(f"unsupported agent action: {type(action).__name__}")

        self._events.extend(emitted)
        return EnvironmentStep(
            observation=self._observation(),
            emitted_events=tuple(emitted),
            done=self._termination is not None,
            termination=self._termination,
            tool_result=tool_results[-1] if tool_results else None,
            tool_results=tuple(tool_results),
        )

    def finish(self, agent_name: str) -> TrialResult:
        task, simulator = self._require_episode()
        termination = self._termination or "incomplete"
        grade = grade_task(goal=task.goal, final_state=simulator.state, events=self._events)
        return TrialResult(
            task_id=task.id,
            agent_name=agent_name,
            termination=termination,
            final_state=deepcopy(simulator.state),
            events=tuple(self._events),
            grade=grade,
            elapsed_ms=(perf_counter() - self._started) * 1000,
        )

    def _observation(self) -> AgentContext:
        if self._task is None:
            raise RuntimeError("reset must be called before observing the environment")
        return AgentContext(
            instruction=self._task.instruction,
            step=self._step,
            events=tuple(self._events),
            episode_id=self._task.id,
        )

    def _require_episode(self) -> tuple[TaskSpec, DeviceSimulator]:
        if self._task is None or self._simulator is None:
            raise RuntimeError("reset must be called before stepping the environment")
        return self._task, self._simulator
