from __future__ import annotations

from typing import Protocol

from device_agent_arena.contracts.engine import ContractDecision, ContractEngine, ContractSession
from device_agent_arena.domain.models import MessageAction, ToolCall, ToolCallBatch
from device_agent_arena.harness.types import AgentContext


class _Agent(Protocol):
    name: str

    def act(self, context: AgentContext) -> ToolCall | ToolCallBatch | MessageAction: ...


class ContractGuardAgent:
    """Apply a schema-bound contract session to every proposal from an inner agent."""

    def __init__(self, inner: _Agent, engine: ContractEngine) -> None:
        self.inner = inner
        self.engine = engine
        self._session: ContractSession | None = None
        self._episode_key: tuple[str, str] | None = None
        self._last_step: int | None = None
        self._telemetry: dict[str, object] | None = None

    @property
    def name(self) -> str:
        return f"contract-guard:{self.inner.name}"

    def act(self, context: AgentContext) -> ToolCall | ToolCallBatch | MessageAction:
        self._ensure_session(context)
        assert self._session is not None
        proposal = self.inner.act(context)
        decision = self._session.intercept(context, proposal)
        self._capture_telemetry(decision)
        self._last_step = context.step
        return decision.action

    def drain_telemetry(self) -> dict[str, object] | None:
        telemetry = self._telemetry
        self._telemetry = None
        return telemetry

    def _ensure_session(self, context: AgentContext) -> None:
        episode_id = context.episode_id
        episode_key = (episode_id, context.instruction)
        legacy_restart = (
            not episode_id
            and self._last_step is not None
            and context.step <= self._last_step
        )
        if self._session is not None and self._episode_key == episode_key and not legacy_restart:
            return
        public_id = episode_id or f"legacy-step-{context.step}"
        self._session = self.engine.start(context.instruction, episode_id=public_id)
        self._episode_key = episode_key
        self._last_step = None

    def _capture_telemetry(self, decision: ContractDecision) -> None:
        drain = getattr(self.inner, "drain_telemetry", None)
        inner_telemetry = drain() if callable(drain) else None
        telemetry = dict(inner_telemetry) if inner_telemetry else {}
        telemetry.update(
            {
                "contract_action": (
                    decision.action.name
                    if isinstance(decision.action, ToolCall)
                    else type(decision.action).__name__
                ),
                "contract_reason": decision.reason,
                "contract_catalog": self.engine.catalog.catalog_id,
            }
        )
        self._telemetry = telemetry
