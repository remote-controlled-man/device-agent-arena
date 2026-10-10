from __future__ import annotations

import re
from copy import deepcopy

from device_agent_arena.agents.model_agent import ModelBackend
from device_agent_arena.domain.models import ToolCall
from device_agent_arena.v1.models import DialogueAction, V1Observation
from device_agent_arena.v1.sft import messages_for_observation

_CLARIFY = re.compile(r"^\s*\[CLARIFY\s+field=([^\]\s]+)\]\s*(.*)$", re.DOTALL)
_CONFIRM = re.compile(r"^\s*\[CONFIRM\s+operation=([^\]]+)\]\s*(.*)$", re.DOTALL)
_REFUSE = re.compile(r"^\s*\[REFUSE\]\s*(.*)$", re.DOTALL)
_ANSWER = re.compile(r"^\s*\[ANSWER\]\s*(.*)$", re.DOTALL)


class V1ModelAgent:
    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend
        self._telemetry: dict | None = None

    @property
    def name(self) -> str:
        return f"v1-model:{self.backend.name}"

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction:
        response = self.backend.complete(
            self._messages(observation),
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": deepcopy(tool.parameters),
                    },
                }
                for tool in observation.tools
            ],
        )
        self._telemetry = dict(response.usage or {})
        if response.raw_text is not None:
            self._telemetry["raw_model_output"] = response.raw_text
        if response.tool_calls:
            self._telemetry["parsed_tool_call_count"] = len(response.tool_calls)
            self._telemetry["ignored_extra_tool_calls"] = max(0, len(response.tool_calls) - 1)
            return response.tool_calls[0]
        return _parse_dialogue(response.text)

    def drain_telemetry(self) -> dict | None:
        telemetry = self._telemetry
        self._telemetry = None
        return telemetry

    def _messages(self, observation: V1Observation) -> list[dict]:
        return messages_for_observation(observation)


def _parse_dialogue(text: str) -> DialogueAction:
    visible = text.strip()
    match = _CLARIFY.match(visible)
    if match:
        return DialogueAction("clarify", match.group(2).strip(), field=match.group(1))
    match = _CONFIRM.match(visible)
    if match:
        return DialogueAction(
            "confirm", match.group(2).strip(), operation=match.group(1).strip()
        )
    match = _REFUSE.match(visible)
    if match:
        return DialogueAction("refuse", match.group(1).strip())
    match = _ANSWER.match(visible)
    if match:
        return DialogueAction("answer", match.group(1).strip())
    if any(marker in visible for marker in ("请问", "请提供", "哪一个", "？", "?")):
        return DialogueAction("clarify", visible, field="device")
    if any(marker in visible for marker in ("拒绝", "不能", "无法", "没有授权")):
        return DialogueAction("refuse", visible)
    return DialogueAction("answer", visible or "任务已完成。")
