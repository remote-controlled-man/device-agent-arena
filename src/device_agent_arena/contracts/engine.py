from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass

from device_agent_arena.contracts.schema import ContractCatalog
from device_agent_arena.domain.models import MessageAction, ToolCall, ToolCallBatch
from device_agent_arena.harness.types import AgentContext

AgentAction = ToolCall | ToolCallBatch | MessageAction


@dataclass(frozen=True)
class ContractDecision:
    action: AgentAction
    reason: str


@dataclass(frozen=True)
class _VisibleObligation:
    operation_id: str
    rule_id: str
    call: ToolCall


class ContractEngine:
    def __init__(self, catalog: ContractCatalog) -> None:
        self.catalog = catalog

    def start(self, instruction: str, *, episode_id: str) -> ContractSession:
        return ContractSession(
            catalog=self.catalog,
            instruction=instruction,
            episode_id=episode_id,
            obligations=_compile_visible_obligations(self.catalog, instruction),
        )


class ContractSession:
    def __init__(
        self,
        *,
        catalog: ContractCatalog,
        instruction: str,
        episode_id: str,
        obligations: tuple[_VisibleObligation, ...],
    ) -> None:
        self.catalog = catalog
        self.instruction = instruction
        self.episode_id = episode_id
        self._obligations = obligations
        self._facts: dict[str, object] = {}
        self._allowed_facts = {
            str(precondition["fact"])
            for operation in catalog.operations
            for precondition in operation.preconditions
        } | {
            str(expression["fact"])
            for operation in catalog.operations
            for effect in operation.effects
            for expression in effect.values()
        }
        self._successful_calls: set[str] = set()
        self._pending: deque[AgentAction] = deque()
        self._handled_result_ids: set[str] = set()
        self._retry_attempts: dict[str, int] = {}

    def intercept(
        self,
        context: AgentContext,
        proposal: AgentAction,
    ) -> ContractDecision:
        self._reduce_public_events(context)
        retry = self._retry_latest_failure(context)
        if retry is not None:
            return retry
        resumed_operation_id: str | None = None
        split_batch_size: int | None = None
        if self._pending:
            proposal = self._pending.popleft()
            if isinstance(proposal, ToolCall):
                try:
                    resumed_operation_id = self.catalog.operation_for_tool(
                        proposal.name
                    ).operation_id
                except KeyError:
                    resumed_operation_id = proposal.name
        elif isinstance(proposal, ToolCallBatch):
            split_batch_size = len(proposal.calls)
            proposal, *remaining = proposal.calls
            self._pending.extend(remaining)
        obligation_reason: str | None = None
        if _is_terminal(proposal, self.catalog):
            obligation = next(
                (
                    item
                    for item in self._obligations
                    if _call_fingerprint(item.call) not in self._successful_calls
                ),
                None,
            )
            if obligation is not None:
                proposal = obligation.call
                obligation_reason = (
                    f"visible_obligation:{obligation.operation_id}:{obligation.rule_id}"
                )
        if not isinstance(proposal, ToolCall):
            return ContractDecision(proposal, "accepted")
        try:
            operation = self.catalog.operation_for_tool(proposal.name)
        except KeyError:
            return ContractDecision(proposal, "unknown_operation")
        for precondition in operation.preconditions:
            fact = str(precondition["fact"])
            expected = precondition.get("equals")
            if fact not in self._facts:
                observer_id = precondition.get("on_unknown", {}).get("operation_id")
                if observer_id is None:
                    continue
                observer = self.catalog.operation(str(observer_id))
                self._pending.appendleft(proposal)
                return ContractDecision(
                    ToolCall(observer.public_name),
                    f"precondition_observe:{operation.operation_id}:{precondition['id']}",
                )
            if self._facts[fact] == expected:
                continue
            repair_id = precondition.get("on_false", {}).get("operation_id")
            if repair_id is None:
                continue
            repair = self.catalog.operation(str(repair_id))
            self._pending.appendleft(proposal)
            return ContractDecision(
                ToolCall(repair.public_name),
                f"precondition_repair:{operation.operation_id}:{precondition['id']}",
            )
        if resumed_operation_id is not None:
            return ContractDecision(
                proposal,
                f"resume_pending:{resumed_operation_id}",
            )
        if split_batch_size is not None:
            return ContractDecision(proposal, f"split_batch:1/{split_batch_size}")
        if obligation_reason is not None:
            return ContractDecision(proposal, obligation_reason)
        return ContractDecision(proposal, "accepted")

    def _retry_latest_failure(self, context: AgentContext) -> ContractDecision | None:
        calls = {
            str(event.payload.get("id")): event
            for event in context.events
            if event.kind == "tool_call"
        }
        for result in reversed(context.events):
            if result.kind != "tool_result" or result.payload.get("ok"):
                continue
            result_id = str(result.payload.get("id"))
            if result_id in self._handled_result_ids:
                continue
            self._handled_result_ids.add(result_id)
            call_event = calls.get(result_id)
            if call_event is None:
                return None
            name = str(call_event.payload.get("name"))
            if name != str(result.payload.get("name")):
                return None
            try:
                operation = self.catalog.operation_for_tool(name)
            except KeyError:
                return None
            code = str(result.payload.get("code"))
            allowed_codes = {str(value) for value in operation.retry.get("allowed_codes", ())}
            max_attempts = int(operation.retry.get("max_attempts", 0))
            arguments = dict(call_event.payload.get("arguments", {}))
            fingerprint = json.dumps(
                [operation.operation_id, arguments],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            attempts = self._retry_attempts.get(fingerprint, 0)
            if not operation.idempotent or code not in allowed_codes or attempts >= max_attempts:
                return None
            attempts += 1
            self._retry_attempts[fingerprint] = attempts
            return ContractDecision(
                ToolCall(name, arguments),
                f"retry:{operation.operation_id}:{code}:{attempts}/{max_attempts}",
            )
        return None

    def _reduce_public_events(self, context: AgentContext) -> None:
        calls = {
            str(event.payload.get("id")): event
            for event in context.events
            if event.kind == "tool_call"
        }
        for event in context.events:
            if event.kind != "tool_result":
                continue
            call_id = str(event.payload.get("id"))
            call_event = calls.get(call_id)
            if call_event is None or str(call_event.payload.get("name")) != str(
                event.payload.get("name")
            ):
                continue
            try:
                operation = self.catalog.operation_for_tool(
                    str(call_event.payload.get("name"))
                )
            except KeyError:
                continue
            if not event.payload.get("ok"):
                if event.payload.get("code") == "precondition_failed":
                    details = event.payload.get("details", {})
                    if isinstance(details, dict):
                        fact = str(details.get("fact", ""))
                        if fact in self._allowed_facts and "actual" in details:
                            self._facts[fact] = details["actual"]
                continue
            completed_call = ToolCall(
                str(call_event.payload.get("name")),
                dict(call_event.payload.get("arguments", {})),
            )
            observation = {
                **dict(event.payload.get("data", {})),
                **dict(event.payload.get("state_delta", {})),
            }
            if _effects_verified(operation.effects, completed_call.arguments, observation):
                self._successful_calls.add(_call_fingerprint(completed_call))
            for source in ("data", "state_delta"):
                self._facts.update(
                    {
                        str(key): value
                        for key, value in dict(event.payload.get(source, {})).items()
                        if str(key) in self._allowed_facts
                    }
                )


def _call_fingerprint(call: ToolCall) -> str:
    return json.dumps(
        [call.name, call.arguments],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _effects_verified(
    effects: tuple[dict[str, object], ...],
    arguments: dict[str, object],
    observation: dict[str, object],
) -> bool:
    for effect in effects:
        if "set" in effect:
            expression = dict(effect["set"])
            expected = expression.get("value")
        else:
            expression = dict(effect["set_from_argument"])
            argument = str(expression.get("argument"))
            if argument not in arguments:
                return False
            expected = arguments[argument]
        fact = str(expression.get("fact"))
        if fact not in observation or observation[fact] != expected:
            return False
    return True


def _is_terminal(action: AgentAction, catalog: ContractCatalog) -> bool:
    if isinstance(action, MessageAction):
        return action.kind in {"answer", "refuse"}
    if isinstance(action, ToolCall):
        try:
            return catalog.operation_for_tool(action.name).role == "terminal"
        except KeyError:
            return False
    return False


def _compile_visible_obligations(
    catalog: ContractCatalog,
    instruction: str,
) -> tuple[_VisibleObligation, ...]:
    normalized = instruction.casefold()
    obligations: list[_VisibleObligation] = []
    for operation in catalog.operations:
        for rule in operation.commitments:
            markers = tuple(str(value).casefold() for value in rule.get("contains_any", ()))
            if markers and not any(marker in normalized for marker in markers):
                continue
            arguments: dict[str, object] = {}
            complete = True
            for name, specification in dict(rule.get("arguments", {})).items():
                mentions = dict(specification.get("enum_mentions", {}))
                matches = [
                    value
                    for value, value_markers in mentions.items()
                    if any(str(marker).casefold() in normalized for marker in value_markers)
                ]
                if len(matches) != 1:
                    complete = False
                    break
                arguments[str(name)] = matches[0]
            if complete:
                obligations.append(
                    _VisibleObligation(
                        operation_id=operation.operation_id,
                        rule_id=str(rule["id"]),
                        call=ToolCall(operation.public_name, arguments),
                    )
                )
    return tuple(obligations)
