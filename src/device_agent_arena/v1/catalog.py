from __future__ import annotations

from dataclasses import dataclass

from device_agent_arena.v1.models import PublicTool


@dataclass(frozen=True)
class ToolAdapter:
    operation: str
    public: PublicTool
    cost: float


def _object_schema(properties: dict, required: tuple[str, ...] = ()) -> dict:
    schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


_OPERATIONS = (
    (
        "search",
        "按用户描述搜索可能的设备；结果只是候选，不保证指向唯一对象。",
        _object_schema({"query": {"type": "string"}}, ("query",)),
        1.0,
    ),
    (
        "inspect_device",
        "读取指定设备的当前状态、归属和共享属性。",
        _object_schema({"device_id": {"type": "string"}}, ("device_id",)),
        1.0,
    ),
    (
        "inspect_routines",
        "查询设备关联的自动化规则及其所有者和活动状态。",
        _object_schema({"device_id": {"type": "string"}}, ("device_id",)),
        1.0,
    ),
    (
        "apply_change",
        "提交设备或自动化变更；返回操作 ID，异步结果必须再查询。",
        _object_schema(
            {
                "action": {
                    "type": "string",
                    "enum": [
                        "set_mode",
                        "schedule_restore",
                        "pause_routine",
                        "wake_device",
                    ],
                },
                "target_id": {"type": "string"},
                "payload": {"type": "object"},
                "idempotency_key": {"type": "string"},
            },
            ("action", "target_id", "payload", "idempotency_key"),
        ),
        2.0,
    ),
    (
        "poll",
        "查询异步操作最终状态；用于确认超时操作是否已提交，或变更是否真正生效。",
        _object_schema({"operation_id": {"type": "string"}}, ("operation_id",)),
        0.5,
    ),
    (
        "rollback",
        "回滚一个已提交且可逆的操作。",
        _object_schema({"operation_id": {"type": "string"}}, ("operation_id",)),
        2.0,
    ),
)

_PROFILE_NAMES = {
    "home-a": {
        "search": "search_devices",
        "inspect_device": "inspect_device",
        "inspect_routines": "inspect_routines",
        "apply_change": "apply_change",
        "poll": "poll_operation",
        "rollback": "rollback_operation",
    },
    "home-b": {
        "search": "find_endpoints",
        "inspect_device": "read_endpoint",
        "inspect_routines": "list_automations",
        "apply_change": "submit_mutation",
        "poll": "check_job",
        "rollback": "undo_job",
    },
    "home-x": {
        "search": "locate_nodes",
        "inspect_device": "describe_node",
        "inspect_routines": "scan_flows",
        "apply_change": "enact_update",
        "poll": "await_update",
        "rollback": "revert_update",
    },
}


def tool_adapters(profile: str) -> tuple[ToolAdapter, ...]:
    try:
        names = _PROFILE_NAMES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown v1 tool profile: {profile}") from exc
    return tuple(
        ToolAdapter(
            operation=operation,
            public=PublicTool(names[operation], description, parameters),
            cost=cost,
        )
        for operation, description, parameters, cost in _OPERATIONS
    )


def public_tool_catalog(profile: str) -> tuple[PublicTool, ...]:
    return tuple(adapter.public for adapter in tool_adapters(profile))


def adapter_for_name(profile: str, name: str) -> ToolAdapter | None:
    return next((item for item in tool_adapters(profile) if item.public.name == name), None)


def public_name_for_operation(profile: str, operation: str) -> str:
    match = next((item for item in tool_adapters(profile) if item.operation == operation), None)
    if match is None:
        raise ValueError(f"unknown v1 operation: {operation}")
    return match.public.name
