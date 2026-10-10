from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from device_agent_arena.domain.models import MessageAction, ToolCall, ToolCallBatch
from device_agent_arena.harness.runner import AgentContext
from device_agent_arena.prompts import BASE_SYSTEM_PROMPT
from device_agent_arena.simulator.tools import tool_schemas


@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, Any] | None = None
    raw_text: str | None = None


class ModelBackend(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse: ...


class ModelAgent:
    def __init__(
        self,
        backend: ModelBackend,
        system_prompt: str = BASE_SYSTEM_PROMPT,
        *,
        tools: list[dict] | None = None,
    ) -> None:
        self.backend = backend
        self.system_prompt = system_prompt
        self.tools = deepcopy(tools) if tools is not None else tool_schemas()
        self._telemetry: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return f"model:{self.backend.name}"

    def act(self, context: AgentContext) -> ToolCall | ToolCallBatch | MessageAction:
        response = self.backend.complete(self._messages(context), deepcopy(self.tools))
        self._telemetry = dict(response.usage) if response.usage is not None else None
        if response.raw_text is not None:
            self._telemetry = self._telemetry or {}
            self._telemetry["raw_model_output"] = response.raw_text
        if response.tool_calls:
            self._telemetry = self._telemetry or {}
            self._telemetry["parsed_tool_call_count"] = len(response.tool_calls)
            self._telemetry["batched_tool_call_count"] = len(response.tool_calls)
            self._telemetry["ignored_extra_tool_calls"] = 0
            if len(response.tool_calls) == 1:
                return response.tool_calls[0]
            return ToolCallBatch(response.tool_calls)

        text = response.text.strip()
        asks_for_detail = any(
            marker in text
            for marker in (
                "请问",
                "请提供",
                "需要提供",
                "请确认",
                "请说明",
                "请告知",
                "请告诉",
                "告诉我",
            )
        ) or any(marker in text for marker in ("？", "?"))
        if asks_for_detail:
            field = (
                "target"
                if any(
                    marker in text
                    for marker in (
                        "谁",
                        "目标",
                        "什么",
                        "对象",
                        "人物",
                        "人员",
                        "姓名",
                        "昵称",
                        "物体",
                        "物品",
                        "target",
                    )
                )
                else None
            )
            return MessageAction(kind="clarify", field=field, content=text)
        if any(marker in text for marker in ("无法", "不能", "不支持", "无关", "不属于", "拒绝")):
            return MessageAction(kind="refuse", content=text)
        return MessageAction(kind="answer", content=text or "操作已完成。")

    def drain_telemetry(self) -> dict[str, Any] | None:
        telemetry = self._telemetry
        self._telemetry = None
        return telemetry

    def _messages(self, context: AgentContext) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context.instruction},
        ]
        rendered_batches: set[str] = set()
        for event in context.events:
            if event.kind == "tool_call":
                batch_id = str(event.payload.get("batch_id", event.payload["id"]))
                if batch_id in rendered_batches:
                    continue
                rendered_batches.add(batch_id)
                batch_events = [
                    candidate
                    for candidate in context.events
                    if candidate.kind in {"tool_call", "tool_result"}
                    and str(
                        candidate.payload.get("batch_id", candidate.payload.get("id"))
                    )
                    == batch_id
                ]
                tool_calls = [
                    candidate for candidate in batch_events if candidate.kind == "tool_call"
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call.payload["id"],
                                "type": "function",
                                "function": {
                                    "name": call.payload["name"],
                                    "arguments": json.dumps(
                                        call.payload.get("arguments", {}), ensure_ascii=False
                                    ),
                                },
                            }
                            for call in tool_calls
                        ],
                    }
                )
                for candidate in batch_events:
                    if candidate.kind == "tool_result":
                        messages.append(
                            {
                                "role": "tool",
                                "name": candidate.payload["name"],
                                "tool_call_id": candidate.payload["id"],
                                "content": json.dumps(
                                    candidate.payload, ensure_ascii=False
                                ),
                            }
                        )
                continue
            if event.kind == "message" and event.payload.get("kind") == "clarify":
                if not event.payload.get("via_tool"):
                    messages.append(
                        {"role": "assistant", "content": event.payload["content"]}
                    )
            elif event.kind == "user" and not event.payload.get("via_tool"):
                messages.append({"role": "user", "content": event.payload["content"]})
        return messages
