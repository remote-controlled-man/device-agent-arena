from __future__ import annotations

from collections import deque

from device_agent_arena.domain.models import MessageAction, ToolCall
from device_agent_arena.harness.runner import AgentContext


class RuleAgent:
    """Small deterministic baseline; intentionally not a hidden-task oracle."""

    name = "rule-v0.1"

    def __init__(self) -> None:
        self._initialized = False
        self._pending: deque[ToolCall | MessageAction] = deque()
        self._retry_after_prerequisite: ToolCall | None = None

    def act(self, context: AgentContext) -> ToolCall | MessageAction:
        if not self._initialized:
            self._initialized = True
            self._pending.extend(self._plan(context.instruction))
            if not self._pending:
                return MessageAction(kind="refuse", content="这不是可执行的设备控制请求。")
            return self._pending.popleft()

        if context.events and context.events[-1].kind == "user":
            field = context.events[-1].payload.get("field")
            content = context.events[-1].payload.get("content")
            if field == "target" and isinstance(content, str):
                kind = "object" if "白板" in content else "person"
                return ToolCall("set_tracking", {"kind": kind, "target": content})

        last_result = next(
            (event for event in reversed(context.events) if event.kind == "tool_result"),
            None,
        )
        last_call_event = next(
            (event for event in reversed(context.events) if event.kind == "tool_call"),
            None,
        )
        last_call = (
            ToolCall(
                name=last_call_event.payload["name"],
                arguments=dict(last_call_event.payload.get("arguments", {})),
            )
            if last_call_event is not None
            else None
        )

        if last_result is not None and not last_result.payload["ok"] and last_call is not None:
            code = last_result.payload["code"]
            message = last_result.payload.get("message", "")
            if code == "transient_failure":
                return last_call
            if code == "precondition_failed" and "not connected" in message:
                self._retry_after_prerequisite = last_call
                return ToolCall(name="connect_device")
            if code == "precondition_failed" and "not calibrated" in message:
                self._retry_after_prerequisite = last_call
                return ToolCall(name="calibrate_device")
            return MessageAction(kind="refuse", content=f"设备操作失败：{message}")

        if self._retry_after_prerequisite is not None:
            retry = self._retry_after_prerequisite
            self._retry_after_prerequisite = None
            return retry
        if self._pending:
            return self._pending.popleft()
        return MessageAction(kind="answer", content="设备操作已完成。")

    @staticmethod
    def _plan(instruction: str) -> tuple[ToolCall | MessageAction, ...]:
        actions: list[ToolCall | MessageAction] = []
        if "横屏" in instruction or "横向" in instruction:
            actions.append(ToolCall("set_orientation", {"orientation": "landscape"}))
        elif "竖屏" in instruction or "竖向" in instruction:
            actions.append(ToolCall("set_orientation", {"orientation": "portrait"}))

        if "演示模式" in instruction:
            actions.append(ToolCall("set_mode", {"mode": "presentation"}))
        elif "正常模式" in instruction:
            actions.append(ToolCall("set_mode", {"mode": "normal"}))
        elif "跟随模式" in instruction:
            actions.append(ToolCall("set_mode", {"mode": "follow"}))

        if "停止" in instruction and "跟踪" in instruction:
            actions.append(ToolCall("set_tracking", {"kind": "off"}))
        elif "跟踪白板" in instruction:
            actions.append(ToolCall("set_tracking", {"kind": "object", "target": "白板"}))
        elif "跟踪演讲者" in instruction:
            actions.append(ToolCall("set_tracking", {"kind": "person", "target": "演讲者"}))
        elif "跟踪" in instruction and "跟随模式" not in instruction:
            actions.append(
                MessageAction(kind="clarify", field="target", content="请问要跟踪谁或什么？")
            )

        if instruction.startswith("恢复") and instruction.endswith("预设"):
            name = instruction.removeprefix("恢复").removesuffix("预设").strip()
            if name:
                actions.append(ToolCall("recall_preset", {"name": name}))
        return tuple(actions)
