from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Literal

from device_agent_arena.agents.hf_backend import normalize_functiongemma_messages
from device_agent_arena.agents.model_agent import ModelBackend
from device_agent_arena.domain.models import ToolCall
from device_agent_arena.v1.models import DialogueAction, PublicTool, V1Observation
from device_agent_arena.v1.sft import messages_for_observation

_CLARIFY = re.compile(r"^\s*\[CLARIFY\s+field=([^\]\s]+)\]\s*(.*)$", re.DOTALL)
_CONFIRM = re.compile(r"^\s*\[CONFIRM\s+operation=([^\]]+)\]\s*(.*)$", re.DOTALL)
_REFUSE = re.compile(r"^\s*\[REFUSE\]\s*(.*)$", re.DOTALL)
_ANSWER = re.compile(r"^\s*\[ANSWER\]\s*(.*)$", re.DOTALL)

_INTERACTION_TOOLS = (
    PublicTool(
        "request_clarification",
        "信息不足或目标不唯一时向用户询问一个必要字段；不得用它代替可获得的工具观测。",
        {
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "需要用户补充的字段"},
                "question": {"type": "string", "description": "向用户展示的具体问题"},
            },
            "required": ["field", "question"],
            "additionalProperties": False,
        },
    ),
    PublicTool(
        "request_confirmation",
        (
            "执行共享或高影响操作前，说明具体影响并请求本轮明确确认。"
            "operation 必须与随后执行的动作和目标一致。"
        ),
        {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "确认键，格式为 action:target_id",
                },
                "message": {"type": "string", "description": "说明影响的确认问题"},
            },
            "required": ["operation", "message"],
            "additionalProperties": False,
        },
    ),
    PublicTool(
        "finish_task",
        "仅在已通过工具结果验证目标完成后结束任务。",
        {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
    ),
    PublicTool(
        "refuse_task",
        "目标不可执行、缺少必要授权或用户拒绝高影响操作时安全结束。",
        {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    ),
)


def _v2_mutation_tool(tool: PublicTool) -> PublicTool:
    parameters = deepcopy(tool.parameters)
    properties = parameters.get("properties", {})
    if "action" not in properties or "idempotency_key" not in properties:
        return PublicTool(tool.name, tool.description, parameters)
    properties["target_id"]["description"] = (
        "动作的真实目标 ID：set_mode/schedule_restore/wake_device 使用 device_id；"
        "pause_routine 使用 routine_id。"
    )
    properties["payload"] = {
        "type": "object",
        "description": (
            "动作参数：set_mode={mode}; schedule_restore={time, mode?}，time 使用 HH:MM；"
            "pause_routine 和 wake_device 使用空对象。字段名是 time，不是 restore_at。"
        ),
        "properties": {
            "mode": {"type": "string", "description": "设备模式"},
            "time": {"type": "string", "description": "恢复时间，HH:MM"},
        },
        "additionalProperties": False,
    }
    properties.pop("idempotency_key", None)
    parameters["required"] = [
        item for item in parameters.get("required", []) if item != "idempotency_key"
    ]
    return PublicTool(
        tool.name,
        (
            f"{tool.description} 先检查目标和共享影响。提交后查询状态。幂等键由运行时协调器管理，"
            "模型不得生成。"
        ),
        parameters,
    )


def v2_public_tools(observation: V1Observation) -> tuple[PublicTool, ...]:
    """Expose a deployment-visible V2 contract without mutating frozen V1 tasks."""
    public = [_v2_mutation_tool(tool) for tool in observation.tools]
    existing = {tool.name for tool in public}
    public.extend(tool for tool in _INTERACTION_TOOLS if tool.name not in existing)
    return tuple(public)


def _tool_schema(tool: PublicTool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": deepcopy(tool.parameters),
        },
    }


def _interaction_action(call: ToolCall) -> DialogueAction | None:
    arguments = call.arguments
    if call.name == "request_clarification":
        return DialogueAction(
            "clarify",
            str(arguments.get("question", "请补充必要信息。")),
            field=str(arguments.get("field", "device")),
        )
    if call.name == "request_confirmation":
        return DialogueAction(
            "confirm",
            str(arguments.get("message", "是否继续？")),
            operation=str(arguments.get("operation", "")),
        )
    if call.name == "finish_task":
        return DialogueAction("answer", str(arguments.get("message", "任务已完成。")))
    if call.name == "refuse_task":
        return DialogueAction("refuse", str(arguments.get("reason", "无法安全执行。")))
    return None


def composition_messages_for_observation(observation: V1Observation) -> list[dict]:
    """Build the exact public prompt used by tool-native composition policies."""
    messages = messages_for_observation(observation)
    messages[0]["content"] += (
        "\nV2 Composition 协议：只使用当前提供的环境原生函数。先观察公开状态和关系；"
        "用户侧动作使用 request_user_action；完成和拒绝分别使用 finish_task 与 refuse_task。"
    )
    return messages


class V2ModelAgent:
    """Model-only V2 adapter with native interaction functions.

    It does not read hidden task state and does not repair model decisions. Contract-aware
    guards are evaluated as a separate compound-agent condition.
    """

    def __init__(
        self,
        backend: ModelBackend,
        *,
        contract_transform: Any | None = None,
        interaction_mode: Literal["dialogue", "tool"] = "dialogue",
    ) -> None:
        if interaction_mode not in {"dialogue", "tool"}:
            raise ValueError("interaction_mode must be 'dialogue' or 'tool'")
        self.backend = backend
        self.contract_transform = contract_transform
        self.interaction_mode = interaction_mode
        self._telemetry: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        suffix = (
            f":{self.contract_transform.profile_id}"
            if self.contract_transform is not None
            else ""
        )
        return f"v2-model:{self.backend.name}{suffix}"

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction:
        if self.interaction_mode == "tool":
            messages = composition_messages_for_observation(observation)
            public_tools = observation.tools
        else:
            messages = messages_for_observation(observation)
            messages[0]["content"] += (
                "\nV2 协议：澄清、确认、完成和拒绝必须使用对应交互函数；不要把它们伪造成设备工具。"
            )
            public_tools = v2_public_tools(observation)
        if self.contract_transform is not None:
            public_tools = self.contract_transform.public_tools(public_tools)
            messages = self.contract_transform.public_messages(messages)
        if bool(getattr(self.backend, "is_functiongemma", False)):
            messages = _native_interaction_history(messages)
        response = self.backend.complete(
            messages,
            [_tool_schema(tool) for tool in public_tools],
        )
        self._telemetry = dict(response.usage or {})
        if response.raw_text is not None:
            self._telemetry["raw_model_output"] = response.raw_text
        if response.tool_calls:
            self._telemetry["parsed_tool_call_count"] = len(response.tool_calls)
            self._telemetry["ignored_extra_tool_calls"] = max(
                0, len(response.tool_calls) - 1
            )
            call = response.tool_calls[0]
            if self.contract_transform is not None:
                call = self.contract_transform.to_runtime_call(call)
            if self.interaction_mode == "tool":
                return call
            return _interaction_action(call) or call
        return DialogueAction("answer", response.text.strip() or "任务已结束。")

    def drain_telemetry(self) -> dict[str, Any] | None:
        telemetry = self._telemetry
        self._telemetry = None
        return telemetry


def _dialogue_call(content: str, call_id: str) -> tuple[dict[str, Any], str] | None:
    match = _CLARIFY.match(content)
    if match:
        name = "request_clarification"
        arguments = {"field": match.group(1), "question": match.group(2).strip()}
    else:
        match = _CONFIRM.match(content)
        if match:
            name = "request_confirmation"
            arguments = {"operation": match.group(1).strip(), "message": match.group(2).strip()}
        else:
            match = _REFUSE.match(content)
            if match:
                name = "refuse_task"
                arguments = {"reason": match.group(1).strip()}
            else:
                match = _ANSWER.match(content)
                if not match:
                    return None
                name = "finish_task"
                arguments = {"message": match.group(1).strip()}
    return (
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        },
        name,
    )


def _v2_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public = []
    for item in deepcopy(tools):
        function = item["function"]
        public.append(
            _v2_mutation_tool(
                PublicTool(
                    name=str(function["name"]),
                    description=str(function.get("description", "")),
                    parameters=dict(function["parameters"]),
                )
            )
        )
    existing = {tool.name for tool in public}
    public.extend(tool for tool in _INTERACTION_TOOLS if tool.name not in existing)
    return [_tool_schema(tool) for tool in public]


def _native_interaction_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    call_index = 0
    pending_interaction: tuple[str, str] | None = None
    converted: list[dict[str, Any]] = []
    for message in messages:
        item = deepcopy(message)
        if item.get("role") == "assistant" and not item.get("tool_calls"):
            call_index += 1
            parsed = _dialogue_call(str(item.get("content", "")), f"interaction_{call_index}")
            if parsed is not None:
                item, name = parsed
                pending_interaction = (item["tool_calls"][0]["id"], name)
        elif item.get("role") == "user" and pending_interaction is not None:
            call_id, name = pending_interaction
            raw = item.get("content", "")
            try:
                response = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                response = raw
            item = {
                "role": "tool",
                "name": name,
                "tool_call_id": call_id,
                "content": json.dumps(response, ensure_ascii=False, sort_keys=True),
            }
            pending_interaction = None
        converted.append(_strip_runtime_fields(item))
    return converted


def _strip_runtime_fields(message: dict[str, Any]) -> dict[str, Any]:
    item = deepcopy(message)
    for call in item.get("tool_calls", []):
        function = call.get("function", {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                parsed.pop("idempotency_key", None)
                function["arguments"] = parsed
        elif isinstance(arguments, dict):
            arguments.pop("idempotency_key", None)
    return item


def functiongemma_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one next-decision record to FunctionGemma's native multi-turn protocol."""
    converted = deepcopy(record)
    messages = _native_interaction_history(
        [*converted["prompt"], *converted["completion"]]
    )
    prompt_length = len(converted["prompt"])
    normalized = normalize_functiongemma_messages(messages)
    converted["prompt"] = normalized[:prompt_length]
    converted["completion"] = normalized[prompt_length:]
    converted["tools"] = _v2_tool_schemas(list(converted.get("tools", [])))
    converted["metadata"] = dict(converted.get("metadata", {}))
    converted["metadata"]["protocol"] = "functiongemma-native-v2"
    return converted
