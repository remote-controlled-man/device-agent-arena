from __future__ import annotations

from dataclasses import dataclass

from device_agent_arena.domain.models import TrajectoryEvent


@dataclass(frozen=True)
class HarnessConfig:
    max_steps: int = 8
    max_consecutive_tool_errors: int = 3
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_consecutive_tool_errors < 1:
            raise ValueError("max_consecutive_tool_errors must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")


@dataclass(frozen=True)
class AgentContext:
    instruction: str
    step: int
    events: tuple[TrajectoryEvent, ...]
    episode_id: str = ""
