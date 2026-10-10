from __future__ import annotations

import re
from collections import deque
from typing import Protocol

from device_agent_arena.domain.models import MessageAction, ToolCall, ToolCallBatch
from device_agent_arena.harness.types import AgentContext


class _Agent(Protocol):
    name: str

    def act(self, context: AgentContext) -> ToolCall | ToolCallBatch | MessageAction: ...


_CONNECTED_TOOLS = {
    "calibrate_device",
    "set_orientation",
    "set_mode",
    "set_tracking",
    "save_preset",
    "recall_preset",
}
_POST_RECALL_MUTATIONS = {"set_mode", "set_orientation", "set_tracking"}
VISIBLE_CONTRACT_GUARD_VERSION = "v0.2"
_INSPECT_MARKERS = (
    "先检查当前设备状态",
    "先检查设备状态",
    "读取设备状态后",
    "先确认设备现状",
)


def _quoted_value(instruction: str, anchor: str) -> str | None:
    match = re.search(rf"{anchor}[^“\"]*[“\"]([^”\"]+)[”\"]", instruction)
    return match.group(1).strip() if match else None


def visible_contract_actions(instruction: str) -> tuple[ToolCall, ...]:
    """Extract only side effects explicitly committed to in the visible instruction."""
    actions: list[ToolCall] = []
    if any(marker in instruction for marker in _INSPECT_MARKERS):
        actions.append(ToolCall("get_device_state"))
    if "未连接" in instruction or "先接入设备" in instruction:
        actions.append(ToolCall("connect_device"))
    if "尚未校准" in instruction or "先完成校准" in instruction:
        actions.append(ToolCall("calibrate_device"))

    recall_name = None
    if "调用预设" in instruction:
        recall_name = _quoted_value(instruction, "调用预设")
    elif "恢复预设" in instruction:
        recall_name = _quoted_value(instruction, "恢复预设")
    if recall_name:
        actions.append(ToolCall("recall_preset", {"name": recall_name}))

    if "演示模式" in instruction:
        actions.append(ToolCall("set_mode", {"mode": "presentation"}))
    elif "跟随模式" in instruction:
        actions.append(ToolCall("set_mode", {"mode": "follow"}))

    if "横屏" in instruction:
        actions.append(ToolCall("set_orientation", {"orientation": "landscape"}))
    elif "竖屏" in instruction:
        actions.append(ToolCall("set_orientation", {"orientation": "portrait"}))

    if "关闭跟踪" in instruction:
        actions.append(ToolCall("set_tracking", {"kind": "off"}))
    else:
        tracking = re.search(r"使用(人物|物体)跟踪([^，。；]+)", instruction)
        if tracking:
            kind_word, target = tracking.groups()
            target = target.strip()
            if target and not target.startswith("指定"):
                actions.append(
                    ToolCall(
                        "set_tracking",
                        {
                            "kind": "person" if kind_word == "人物" else "object",
                            "target": target,
                        },
                    )
                )

    save_name = _quoted_value(instruction, "保存为") if "保存为" in instruction else None
    if save_name:
        actions.append(ToolCall("save_preset", {"name": save_name}))
    return tuple(actions)


class VisibleContractGuardAgent:
    """Guard an agent using only its visible instruction and observed tool results."""

    def __init__(self, inner: _Agent) -> None:
        self.inner = inner
        self._instruction: str | None = None
        self._pending: deque[ToolCall | ToolCallBatch | MessageAction] = deque()
        self._known_state: dict[str, object] = {}
        self._successful_tools: set[str] = set()
        self._handled_error_ids: set[str] = set()
        self._telemetry: dict[str, object] | None = None

    @property
    def name(self) -> str:
        return f"visible-contract-guard:{self.inner.name}"

    def act(self, context: AgentContext) -> ToolCall | ToolCallBatch | MessageAction:
        self._reset_if_needed(context)
        self._sync_observations(context)

        recovery = self._recover_latest_error(context)
        if recovery is not None:
            self._telemetry = {"guard_action": recovery.name, "guard_reason": "tool_recovery"}
            return self._guard_tool_call(recovery)

        if self._pending:
            candidate = self._pending.popleft()
            self._telemetry = {"guard_reason": "queued_contract_action"}
        else:
            candidate = self.inner.act(context)
            drain = getattr(self.inner, "drain_telemetry", None)
            inner_telemetry = drain() if callable(drain) else None
            self._telemetry = dict(inner_telemetry) if inner_telemetry else {}

        if isinstance(candidate, ToolCall):
            post_recall = self._replace_uncommitted_post_recall(candidate)
            if post_recall is not candidate:
                self._record_override(post_recall, "uncommitted_post_recall")
                candidate = post_recall
            if candidate.name == "finish_task":
                replacement = self._defer_terminal(candidate)
                if replacement is not candidate:
                    self._record_override(replacement, "visible_contract")
                candidate = replacement
            return self._guard_tool_call(candidate)
        if isinstance(candidate, MessageAction) and candidate.kind in {"answer", "refuse"}:
            replacement = self._defer_terminal(candidate)
            if replacement is not candidate:
                self._record_override(replacement, "visible_contract")
                return self._guard_tool_call(replacement)
        return candidate

    def drain_telemetry(self) -> dict[str, object] | None:
        telemetry = self._telemetry
        self._telemetry = None
        return telemetry

    def _reset_if_needed(self, context: AgentContext) -> None:
        if self._instruction == context.instruction:
            return
        self._instruction = context.instruction
        self._pending.clear()
        self._known_state = {}
        self._successful_tools.clear()
        self._handled_error_ids.clear()
        self._telemetry = None
        if "未连接" in context.instruction:
            self._known_state["connected"] = False
            self._known_state["calibrated"] = False
        elif "已连接但尚未校准" in context.instruction:
            self._known_state["connected"] = True
            self._known_state["calibrated"] = False
        elif "设备已在线" in context.instruction or "当前在线设备" in context.instruction:
            self._known_state["connected"] = True

    def _sync_observations(self, context: AgentContext) -> None:
        for event in context.events:
            if event.kind != "tool_result" or not event.payload.get("ok"):
                continue
            name = str(event.payload["name"])
            self._successful_tools.add(name)
            if name == "get_device_state":
                self._known_state.update(dict(event.payload.get("data", {})))
            self._known_state.update(dict(event.payload.get("state_delta", {})))

    def _recover_latest_error(self, context: AgentContext) -> ToolCall | None:
        result = next(
            (
                event
                for event in reversed(context.events)
                if event.kind == "tool_result"
                and not event.payload.get("ok")
                and str(event.payload.get("id")) not in self._handled_error_ids
            ),
            None,
        )
        if result is None:
            return None
        result_id = str(result.payload.get("id"))
        self._handled_error_ids.add(result_id)
        call_event = next(
            (
                event
                for event in reversed(context.events)
                if event.kind == "tool_call" and str(event.payload.get("id")) == result_id
            ),
            None,
        )
        if call_event is None:
            return None
        failed_call = ToolCall(
            str(call_event.payload["name"]),
            dict(call_event.payload.get("arguments", {})),
        )
        code = str(result.payload.get("code", ""))
        message = str(result.payload.get("message", ""))
        if code == "transient_failure":
            return failed_call
        if code == "precondition_failed" and "not connected" in message:
            self._known_state["connected"] = False
            self._known_state["calibrated"] = False
            self._pending.appendleft(failed_call)
            return ToolCall("connect_device")
        if code == "precondition_failed" and "not calibrated" in message:
            self._known_state["calibrated"] = False
            self._pending.appendleft(failed_call)
            return ToolCall("calibrate_device")
        return None

    def _defer_terminal(
        self, terminal: ToolCall | MessageAction
    ) -> ToolCall | MessageAction:
        assert self._instruction is not None
        for obligation in visible_contract_actions(self._instruction):
            if obligation.name not in self._successful_tools:
                self._pending.appendleft(terminal)
                return obligation
        return terminal

    def _replace_uncommitted_post_recall(self, candidate: ToolCall) -> ToolCall:
        if (
            "recall_preset" not in self._successful_tools
            or candidate.name not in _POST_RECALL_MUTATIONS
        ):
            return candidate
        assert self._instruction is not None
        contract = visible_contract_actions(self._instruction)
        committed_names = {action.name for action in contract}
        if candidate.name in committed_names:
            return candidate
        for obligation in contract:
            if obligation.name not in self._successful_tools:
                return obligation
        return ToolCall(
            "finish_task",
            {"kind": "answer", "content": "显式设备配置已完成。"},
        )

    def _guard_tool_call(self, candidate: ToolCall) -> ToolCall:
        connected = self._known_state.get("connected")
        if candidate.name in _CONNECTED_TOOLS and connected is False:
            self._pending.appendleft(candidate)
            replacement = ToolCall("connect_device")
            self._record_override(replacement, "known_disconnected")
            return replacement

        requires_calibration = (
            candidate.name == "set_mode" and candidate.arguments.get("mode") == "follow"
        ) or (
            candidate.name == "set_tracking" and candidate.arguments.get("kind") != "off"
        )
        if requires_calibration and self._known_state.get("calibrated") is False:
            self._pending.appendleft(candidate)
            replacement = ToolCall("calibrate_device")
            self._record_override(replacement, "known_uncalibrated")
            return replacement
        return candidate

    def _record_override(self, replacement: ToolCall, reason: str) -> None:
        self._telemetry = self._telemetry or {}
        self._telemetry["guard_action"] = replacement.name
        self._telemetry["guard_reason"] = reason
