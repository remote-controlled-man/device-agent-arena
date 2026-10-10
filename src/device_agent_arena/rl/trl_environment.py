from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import Any

from device_agent_arena.domain.models import TaskSpec, ToolCall, TrialResult
from device_agent_arena.harness.environment import AgentEnvironment, EnvironmentStep
from device_agent_arena.harness.types import HarnessConfig
from device_agent_arena.rl.rewards import REWARD_MODEL_VERSION, RewardModel, RewardTrace


class DeviceGRPOEnvironment:
    """Adapt the benchmark MDP to TRL's stateful ``environment_factory`` contract."""

    def __init__(
        self,
        tasks: Iterable[TaskSpec],
        *,
        max_steps: int = 10,
        reward_model_version: str = REWARD_MODEL_VERSION,
        rollout_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._tasks = {task.id: task for task in tasks}
        if not self._tasks:
            raise ValueError("DeviceGRPOEnvironment requires at least one task")
        self._environment = AgentEnvironment(HarnessConfig(max_steps=max_steps))
        self._task: TaskSpec | None = None
        self._reward_model = RewardModel(version=reward_model_version)
        self._rollout_sink = rollout_sink
        self._reward_trace: RewardTrace | None = None
        self._last_trial: TrialResult | None = None
        self._last_rollout_record: dict[str, Any] | None = None

    @property
    def last_trial(self) -> TrialResult | None:
        return self._last_trial

    @property
    def last_reward_trace(self) -> RewardTrace | None:
        return self._reward_trace

    def reset(self, task_id: str, **_: Any) -> str:
        """Reset one deterministic task selected by the training dataset."""
        try:
            task = self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown RL task: {task_id}") from exc
        self._task = task
        self._environment.reset(task)
        self._reward_trace = None
        self._last_trial = None
        self._last_rollout_record = None
        return task.instruction

    def get_reward(self) -> float:
        """Return the deterministic environment reward for the completed rollout."""
        if self._reward_trace is not None:
            return self._reward_trace.total
        task = self._require_task()
        trial = self._environment.finish("trl-grpo-policy")
        self._last_trial = trial
        self._reward_trace = self._reward_model.score(task, trial)
        self._last_rollout_record = {
            "task_id": task.id,
            "task_seed": task.seed,
            "task_structure": task.provenance.get("structure"),
            "trial": asdict(trial),
            "reward_trace": asdict(self._reward_trace),
        }
        if self._rollout_sink is not None:
            self._rollout_sink(self._last_rollout_record)
        return self._reward_trace.total

    def _annotate_termination_mask(self, payload: dict[str, Any]) -> None:
        if self._last_rollout_record is None:
            raise RuntimeError("get_reward must run before termination-mask annotation")
        self._last_rollout_record["termination_mask"] = dict(payload)

    def get_device_state(self) -> str:
        """Read the observable device state.

        Returns:
            A JSON tool result containing the current observable state.
        """
        return self._execute_tool("get_device_state")

    def connect_device(self) -> str:
        """Connect the device before operations that require a connection.

        Returns:
            A JSON tool result describing whether the connection succeeded.
        """
        return self._execute_tool("connect_device")

    def calibrate_device(self) -> str:
        """Calibrate a connected device before enabling tracking.

        Returns:
            A JSON tool result describing whether calibration succeeded.
        """
        return self._execute_tool("calibrate_device")

    def set_mode(self, mode: str) -> str:
        """Set the operating mode.

        Args:
            mode: One of ``normal``, ``presentation``, or ``follow``.

        Returns:
            A JSON tool result describing the mode transition.
        """
        return self._execute_tool("set_mode", mode=mode)

    def set_orientation(self, orientation: str) -> str:
        """Set the device orientation.

        Args:
            orientation: Either ``portrait`` or ``landscape``.

        Returns:
            A JSON tool result describing the orientation transition.
        """
        return self._execute_tool("set_orientation", orientation=orientation)

    def set_tracking(self, kind: str, target: str | None = None) -> str:
        """Enable or disable tracking for a target.

        Args:
            kind: One of ``off``, ``person``, or ``object``.
            target: Target name when tracking is enabled.

        Returns:
            A JSON tool result describing the tracking transition.
        """
        arguments = {"kind": kind}
        if target is not None:
            arguments["target"] = target
        return self._execute_tool("set_tracking", **arguments)

    def save_preset(self, name: str) -> str:
        """Save the current device configuration as a named preset.

        Args:
            name: Non-empty preset name.

        Returns:
            A JSON tool result describing whether the preset was saved.
        """
        return self._execute_tool("save_preset", name=name)

    def recall_preset(self, name: str) -> str:
        """Recall a previously saved preset.

        Args:
            name: Existing preset name.

        Returns:
            A JSON tool result describing whether the preset was recalled.
        """
        return self._execute_tool("recall_preset", name=name)

    def request_clarification(self, field: str, content: str) -> str:
        """Ask the user for one missing task parameter.

        Args:
            field: Name of the missing field, such as ``target``.
            content: The concise clarification question shown to the user.

        Returns:
            A JSON object containing the user's deterministic clarification.
        """
        return self._execute_tool(
            "request_clarification",
            field=field,
            content=content,
        )

    def finish_task(self, kind: str, content: str) -> str:
        """Finish with an answer or refusal after the environment goal is resolved.

        Args:
            kind: Either ``answer`` or ``refuse``.
            content: Concise final response to the user.

        Returns:
            A JSON tool result confirming termination.
        """
        return self._execute_tool("finish_task", kind=kind, content=content)

    def _execute_tool(self, tool_name: str, **arguments: Any) -> str:
        step = self._environment.step(ToolCall(name=tool_name, arguments=arguments))
        return self._serialize_step(step)

    @staticmethod
    def _serialize_step(step: EnvironmentStep) -> str:
        if step.tool_result is None:
            raise RuntimeError("device tool transition did not produce a ToolResult")
        payload = {
            **asdict(step.tool_result),
            "done": step.done,
            "termination": step.termination,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _require_task(self) -> TaskSpec:
        if self._task is None:
            raise RuntimeError("reset must be called before scoring the environment")
        return self._task
