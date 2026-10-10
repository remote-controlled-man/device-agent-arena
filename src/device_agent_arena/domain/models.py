from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Orientation = Literal["portrait", "landscape"]
TrackingKind = Literal["off", "person", "object"]
DeviceMode = Literal["normal", "presentation", "follow"]


@dataclass
class DeviceState:
    connected: bool = False
    calibrated: bool = False
    orientation: Orientation = "portrait"
    tracking: TrackingKind = "off"
    mode: DeviceMode = "normal"
    target: str | None = None
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallBatch:
    """Ordered tool calls emitted by one assistant generation."""

    calls: tuple[ToolCall, ...]

    def __post_init__(self) -> None:
        if len(self.calls) < 2:
            raise ValueError("ToolCallBatch requires at least two calls")


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    code: str
    message: str
    state_delta: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FaultSpec:
    tool: str
    invocation: int = 1
    code: str = "transient_failure"
    probability: float = 1.0


@dataclass(frozen=True)
class GoalSpec:
    final_state: dict[str, Any] = field(default_factory=dict)
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    required_clarifications: tuple[str, ...] = ()
    terminal_behavior: Literal["answer", "refuse"] = "answer"


@dataclass(frozen=True)
class TaskSpec:
    id: str
    version: str
    split: Literal["train", "dev", "test"]
    categories: tuple[str, ...]
    seed: int
    instruction: str
    reference_instruction: str
    initial_state: DeviceState
    fault_plan: tuple[FaultSpec, ...]
    clarifications: dict[str, str]
    goal: GoalSpec
    provenance: dict[str, str]


@dataclass(frozen=True)
class TrajectoryEvent:
    step: int
    kind: str
    payload: dict[str, Any]
    elapsed_ms: float | None = None


@dataclass(frozen=True)
class GradeResult:
    success: bool
    checks: dict[str, bool]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MessageAction:
    kind: Literal["answer", "clarify", "refuse"]
    content: str
    field: str | None = None


@dataclass(frozen=True)
class TrialResult:
    task_id: str
    agent_name: str
    termination: str
    final_state: DeviceState
    events: tuple[TrajectoryEvent, ...]
    grade: GradeResult
    elapsed_ms: float
