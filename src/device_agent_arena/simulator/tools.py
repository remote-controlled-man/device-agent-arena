from __future__ import annotations

from copy import deepcopy

_DEVICE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_device_state",
            "description": "读取设备当前连接、校准、屏幕方向、模式、跟踪和预设状态。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "connect_device",
            "description": "连接设备。设备未连接导致操作失败时调用。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calibrate_device",
            "description": "校准已连接设备。跟随模式和目标跟踪需要先完成校准。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_orientation",
            "description": "设置设备屏幕方向。",
            "parameters": {
                "type": "object",
                "properties": {
                    "orientation": {"type": "string", "enum": ["portrait", "landscape"]}
                },
                "required": ["orientation"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_mode",
            "description": "设置设备工作模式。follow 模式要求设备已校准。",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["normal", "presentation", "follow"],
                    }
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_tracking",
            "description": "开启或停止人物/物体跟踪；开启时必须提供 target。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["off", "person", "object"]},
                    "target": {"type": "string"},
                },
                "required": ["kind"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_preset",
            "description": "将当前设备配置保存为命名预设。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "minLength": 1}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_preset",
            "description": "恢复一个已存在的命名预设。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "minLength": 1}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
]

_INTERACTION_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "request_clarification",
            "description": "参数缺失时向用户询问一个明确字段；收到工具结果后再继续执行。",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": ["target"]},
                    "content": {"type": "string", "minLength": 1},
                },
                "required": ["field", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": "仅在任务完成或应拒绝请求时结束本轮任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["answer", "refuse"]},
                    "content": {"type": "string", "minLength": 1},
                },
                "required": ["kind", "content"],
                "additionalProperties": False,
            },
        },
    },
]


def tool_schemas(*, include_interaction: bool = True) -> list[dict]:
    tools = [*_DEVICE_TOOLS, *_INTERACTION_TOOLS] if include_interaction else _DEVICE_TOOLS
    return deepcopy(tools)


TOOL_NAMES = frozenset(tool["function"]["name"] for tool in _DEVICE_TOOLS)
INTERACTION_TOOL_NAMES = frozenset(
    tool["function"]["name"] for tool in _INTERACTION_TOOLS
)
_PARAMETERS_BY_TOOL = {
    tool["function"]["name"]: tool["function"]["parameters"]
    for tool in (*_DEVICE_TOOLS, *_INTERACTION_TOOLS)
}


def validate_tool_arguments(name: str, arguments: object) -> str | None:
    """Return a stable validation error, or ``None`` when a call matches its schema."""
    parameters = _PARAMETERS_BY_TOOL.get(name)
    if parameters is None:
        return "unknown tool"
    if not isinstance(arguments, dict):
        return "arguments must be an object"

    properties = parameters.get("properties", {})
    extras = sorted(set(arguments) - set(properties))
    if extras:
        return f"unexpected arguments: {extras}"
    missing = sorted(set(parameters.get("required", ())) - set(arguments))
    if missing:
        return f"missing required arguments: {missing}"

    for key, value in arguments.items():
        specification = properties[key]
        if specification.get("type") == "string" and not isinstance(value, str):
            return f"{key} must be a string"
        if "enum" in specification and value not in specification["enum"]:
            return f"{key} must be one of {specification['enum']}"
        if specification.get("minLength") and isinstance(value, str) and not value.strip():
            return f"{key} must be non-empty"
    return None
