from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from device_agent_arena.domain.models import TrajectoryEvent

Split = Literal["train", "dev", "test"]
DialogueKind = Literal["answer", "clarify", "confirm", "refuse"]


@dataclass(frozen=True)
class PublicTool:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class V1FaultSpec:
    action: str
    invocation: int
    kind: Literal["timeout_after_commit", "dropped_write", "async_failure"]


@dataclass(frozen=True)
class V1DynamicSpec:
    trigger_operation: str
    invocation: int
    changes: dict[str, Any]
    notification: str


@dataclass(frozen=True)
class V1SafetyPolicy:
    high_impact_actions: tuple[str, ...] = ()
    shared_change_requires_confirmation: bool = True
    policy_text: tuple[str, ...] = ()


@dataclass(frozen=True)
class V1Goal:
    state_requirements: dict[str, Any] = field(default_factory=dict)
    required_tools: tuple[str, ...] = ()
    required_clarifications: tuple[str, ...] = ()
    required_confirmations: tuple[str, ...] = ()
    allowed_mutation_targets: tuple[str, ...] = ()
    forbidden_mutations: tuple[str, ...] = ()
    terminal_behavior: Literal["answer", "refuse"] = "answer"
    max_cost: float = 20.0


@dataclass(frozen=True)
class PlanStep:
    kind: Literal["tool", "clarify", "confirm", "answer", "refuse"]
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    field: str | None = None
    operation: str | None = None
    content: str = ""


@dataclass(frozen=True)
class V1Observation:
    episode_id: str
    instruction: str
    public_context: dict[str, Any]
    safety_policy: tuple[str, ...]
    tools: tuple[PublicTool, ...]
    step: int
    events: tuple[TrajectoryEvent, ...]


@dataclass(frozen=True)
class DialogueAction:
    kind: DialogueKind
    content: str
    field: str | None = None
    operation: str | None = None


@dataclass(frozen=True)
class V1TaskSpec:
    id: str
    version: str
    split: Split
    categories: tuple[str, ...]
    seed: int
    instruction: str
    public_context: dict[str, Any]
    tool_profile: str
    initial_world: dict[str, Any]
    user_script: dict[str, str]
    safety_policy: V1SafetyPolicy
    fault_plan: tuple[V1FaultSpec, ...]
    dynamic_plan: tuple[V1DynamicSpec, ...]
    goal: V1Goal
    oracle_plan: tuple[PlanStep, ...]
    provenance: dict[str, str]

    def public_observation(
        self,
        *,
        step: int,
        events: tuple[TrajectoryEvent, ...],
    ) -> V1Observation:
        # Import locally so the domain model does not own catalog implementations.
        from device_agent_arena.v1.catalog import public_tool_catalog

        return V1Observation(
            episode_id=self.id,
            instruction=self.instruction,
            public_context=dict(self.public_context),
            safety_policy=self.safety_policy.policy_text,
            tools=public_tool_catalog(self.tool_profile),
            step=step,
            events=events,
        )


@dataclass(frozen=True)
class V1GradeResult:
    success: bool
    goal_completion: bool
    safe_execution: bool
    checks: dict[str, bool]
    violations: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class V1TrialResult:
    task_id: str
    agent_name: str
    termination: str
    final_world: dict[str, Any]
    events: tuple[TrajectoryEvent, ...]
    grade: V1GradeResult
    elapsed_ms: float
