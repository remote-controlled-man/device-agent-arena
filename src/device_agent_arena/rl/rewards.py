from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass

from device_agent_arena.domain.models import (
    MessageAction,
    TaskSpec,
    ToolCall,
    TrajectoryEvent,
    TrialResult,
)
from device_agent_arena.grading.task_grader import grade_task
from device_agent_arena.simulator.device import DeviceSimulator
from device_agent_arena.simulator.tools import INTERACTION_TOOL_NAMES

REWARD_MODEL_VERSION = "0.3"


@dataclass(frozen=True)
class RewardWeights:
    terminal_success: float = 1.0
    goal_progress: float = 0.05
    valid_action: float = 0.02
    recovery: float = 0.1
    invalid_action: float = -0.2
    redundant_action: float = -0.05
    forbidden_side_effect: float = -0.5
    missing_required_clarification: float = -0.5
    step_cost: float = -0.01
    failure: float = -1.0
    shaping_budget: float = 0.1
    repeated_action_failure: float = -0.1


@dataclass(frozen=True)
class StepReward:
    step: int
    components: dict[str, float]
    total: float


@dataclass(frozen=True)
class RewardTrace:
    components: dict[str, float]
    steps: tuple[StepReward, ...]
    total: float


_PROGRESS_PREFIXES = ("final_state.", "required_tool.", "required_clarification.")
_RECOVERABLE_CODES = {"precondition_failed", "transient_failure"}


class RewardModel:
    """Deterministically recompute shaped rewards from a recorded trial."""

    def __init__(
        self,
        weights: RewardWeights | None = None,
        *,
        version: str = REWARD_MODEL_VERSION,
    ) -> None:
        if version != REWARD_MODEL_VERSION:
            raise ValueError(f"unsupported reward model version: {version}")
        self.version = version
        self.weights = weights or RewardWeights()

    def score(self, task: TaskSpec, trial: TrialResult) -> RewardTrace:
        if trial.task_id != task.id:
            raise ValueError("trial task_id does not match the reward task")

        simulator = DeviceSimulator(
            deepcopy(task.initial_state),
            task.fault_plan,
            seed=task.seed,
        )
        prefix: list[TrajectoryEvent] = []
        by_step: dict[int, list[TrajectoryEvent]] = defaultdict(list)
        for event in trial.events:
            by_step[event.step].append(event)

        step_rewards: list[StepReward] = []
        previous_tool_code: str | None = None
        previous_tool_name: str | None = None
        seen_clarifications: set[str | None] = set()
        seen_state_reads: set[str] = set()
        for step, events in sorted(by_step.items()):
            action_events = [
                event
                for event in events
                if event.kind == "message"
                or (
                    event.kind == "tool_call"
                    and event.payload.get("name") not in INTERACTION_TOOL_NAMES
                )
            ]
            if not action_events:
                prefix.extend(events)
                continue

            before_grade = grade_task(
                goal=task.goal,
                final_state=simulator.state,
                events=prefix,
            )
            components: dict[str, float] = {"step_cost": self.weights.step_cost}

            for action_event in action_events:
                if action_event.kind == "tool_call":
                    before_state = asdict(simulator.state)
                    tool_name = str(action_event.payload.get("name"))
                    arguments = action_event.payload.get("arguments", {})
                    result = simulator.execute(
                        ToolCall(name=tool_name, arguments=dict(arguments))
                    )
                    call_id = action_event.payload.get("id")
                    recorded_result = next(
                        (
                            event
                            for event in events
                            if event.kind == "tool_result"
                            and event.payload.get("id") == call_id
                        ),
                        None,
                    )
                    if recorded_result is None:
                        raise ValueError(f"step {step} is missing tool result {call_id}")
                    if (
                        recorded_result.payload.get("name") != tool_name
                        or recorded_result.payload.get("ok") is not result.ok
                        or recorded_result.payload.get("code") != result.code
                    ):
                        raise ValueError(
                            f"step {step} tool result does not replay deterministically"
                        )

                    if result.code in {"invalid_arguments", "unknown_tool"}:
                        self._add_component(
                            components, "invalid_action", self.weights.invalid_action
                        )
                    else:
                        self._add_component(
                            components, "valid_action", self.weights.valid_action
                        )
                    after_state = asdict(simulator.state)
                    if (
                        result.ok
                        and before_state == after_state
                        and (
                            tool_name != "get_device_state"
                            or json.dumps(before_state, ensure_ascii=False, sort_keys=True)
                            in seen_state_reads
                        )
                    ):
                        self._add_component(
                            components, "redundant_action", self.weights.redundant_action
                        )
                    if tool_name == "get_device_state" and result.ok:
                        seen_state_reads.add(
                            json.dumps(before_state, ensure_ascii=False, sort_keys=True)
                        )
                    if tool_name in task.goal.forbidden_tools:
                        self._add_component(
                            components,
                            "forbidden_side_effect",
                            self.weights.forbidden_side_effect,
                        )
                    if (
                        previous_tool_code in _RECOVERABLE_CODES
                        and result.ok
                        and (tool_name == previous_tool_name or before_state != after_state)
                    ):
                        self._add_component(
                            components, "recovery", self.weights.recovery
                        )
                    previous_tool_code = result.code
                    previous_tool_name = tool_name
                else:
                    message = MessageAction(
                        kind=action_event.payload["kind"],
                        content=action_event.payload.get("content", ""),
                        field=action_event.payload.get("field"),
                    )
                    if message.kind == "clarify":
                        if message.field not in task.clarifications:
                            self._add_component(
                                components, "invalid_action", self.weights.invalid_action
                            )
                        elif message.field in seen_clarifications:
                            self._add_component(
                                components,
                                "redundant_action",
                                self.weights.redundant_action,
                            )
                        seen_clarifications.add(message.field)

            prefix.extend(events)
            after_grade = grade_task(
                goal=task.goal,
                final_state=simulator.state,
                events=prefix,
            )
            newly_satisfied = sum(
                not before_grade.checks.get(name, False) and satisfied
                for name, satisfied in after_grade.checks.items()
                if name.startswith(_PROGRESS_PREFIXES)
            )
            if newly_satisfied:
                components["goal_progress"] = self.weights.goal_progress * newly_satisfied
            step_rewards.append(self._step_reward(step, components))

        if not step_rewards:
            replayed_grade = grade_task(
                goal=task.goal,
                final_state=simulator.state,
                events=prefix,
            )
            if asdict(simulator.state) != asdict(trial.final_state):
                raise ValueError("trial final state does not match deterministic replay")
            if json.dumps(asdict(replayed_grade), sort_keys=True) != json.dumps(
                asdict(trial.grade), sort_keys=True
            ):
                raise ValueError("trial grade does not match deterministic replay")
            components = {"failure": self.weights.failure}
            missing_clarifications = sum(
                not satisfied
                for name, satisfied in replayed_grade.checks.items()
                if name.startswith("required_clarification.")
            )
            if missing_clarifications:
                raw_missing = (
                    self.weights.missing_required_clarification * missing_clarifications
                )
                components["missing_required_clarification"] = raw_missing
                bounded = max(-self.weights.shaping_budget, raw_missing)
                components["shaping_cap_adjustment"] = bounded - raw_missing
            step_reward = self._step_reward(0, components)
            return RewardTrace(
                components=step_reward.components,
                steps=(step_reward,),
                total=step_reward.total,
            )
        replayed_grade = grade_task(
            goal=task.goal,
            final_state=simulator.state,
            events=prefix,
        )
        if asdict(simulator.state) != asdict(trial.final_state):
            raise ValueError("trial final state does not match deterministic replay")
        if json.dumps(asdict(replayed_grade), sort_keys=True) != json.dumps(
            asdict(trial.grade), sort_keys=True
        ):
            raise ValueError("trial grade does not match deterministic replay")
        if not replayed_grade.success:
            step_rewards = [
                self._step_reward(
                    step.step,
                    {name: value for name, value in step.components.items() if name != "recovery"},
                )
                for step in step_rewards
            ]
        terminal_name = "terminal_success" if replayed_grade.success else "failure"
        terminal_value = getattr(self.weights, terminal_name)
        final_step = step_rewards[-1]
        final_components = {**final_step.components, terminal_name: terminal_value}
        missing_clarifications = sum(
            not satisfied
            for name, satisfied in replayed_grade.checks.items()
            if name.startswith("required_clarification.")
        )
        if missing_clarifications:
            final_components["missing_required_clarification"] = (
                self.weights.missing_required_clarification * missing_clarifications
            )
        shaping_total = sum(
            sum(
                value
                for name, value in step.components.items()
                if name not in {"terminal_success", "failure"}
            )
            for step in (*step_rewards[:-1], self._step_reward(final_step.step, final_components))
        )
        bounded_shaping = max(
            -self.weights.shaping_budget,
            min(self.weights.shaping_budget, shaping_total),
        )
        shaping_adjustment = bounded_shaping - shaping_total
        if shaping_adjustment:
            final_components["shaping_cap_adjustment"] = shaping_adjustment
        if not replayed_grade.success and any(
            "redundant_action" in step.components for step in step_rewards
        ):
            final_components["repeated_action_failure"] = self.weights.repeated_action_failure
        step_rewards[-1] = self._step_reward(final_step.step, final_components)

        aggregate: dict[str, float] = defaultdict(float)
        for step_reward in step_rewards:
            for name, value in step_reward.components.items():
                aggregate[name] += value
        components = {name: round(value, 6) for name, value in aggregate.items() if value}
        return RewardTrace(
            components=components,
            steps=tuple(step_rewards),
            total=round(sum(step.total for step in step_rewards), 6),
        )

    @staticmethod
    def _step_reward(step: int, components: dict[str, float]) -> StepReward:
        cleaned = {name: round(value, 6) for name, value in components.items() if value}
        return StepReward(step=step, components=cleaned, total=round(sum(cleaned.values()), 6))

    @staticmethod
    def _add_component(components: dict[str, float], name: str, value: float) -> None:
        components[name] = components.get(name, 0.0) + value
