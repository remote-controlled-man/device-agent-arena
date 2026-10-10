from __future__ import annotations

from typing import Protocol

from device_agent_arena.domain.models import (
    MessageAction,
    TaskSpec,
    ToolCall,
    ToolCallBatch,
    TrialResult,
)
from device_agent_arena.harness.environment import AgentEnvironment
from device_agent_arena.harness.types import AgentContext, HarnessConfig


class Agent(Protocol):
    name: str

    def act(self, context: AgentContext) -> ToolCall | ToolCallBatch | MessageAction: ...


class EvaluationHarness:
    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig()

    def run(self, task: TaskSpec, agent: Agent) -> TrialResult:
        environment = AgentEnvironment(self.config)
        observation = environment.reset(task)
        while True:
            action = agent.act(observation)
            drain_telemetry = getattr(agent, "drain_telemetry", None)
            telemetry = drain_telemetry() if callable(drain_telemetry) else None
            transition = environment.step(action, model_event=telemetry)
            if transition.done:
                return environment.finish(agent.name)
            observation = transition.observation
