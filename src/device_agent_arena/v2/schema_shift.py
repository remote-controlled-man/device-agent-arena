from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from device_agent_arena.domain.models import ToolCall
from device_agent_arena.v1.models import PublicTool

_KNOWN_NAMES = {
    "search": ("search_devices", "find_endpoints", "locate_nodes"),
    "inspect_device": ("inspect_device", "read_endpoint", "describe_node"),
    "inspect_routines": ("inspect_routines", "list_automations", "scan_flows"),
    "apply_change": ("apply_change", "submit_mutation", "enact_update"),
    "poll": ("poll_operation", "check_job", "await_update"),
    "rollback": ("rollback_operation", "undo_job", "revert_update"),
}

_PROFILES = {
    "identity": {
        "names": {},
        "arguments": {},
        "payload": {},
        "distractors": 0,
    },
    "schema-a": {
        "names": {
            "search": "discover_assets",
            "inspect_device": "read_asset",
            "inspect_routines": "list_dependencies",
            "apply_change": "commit_asset_change",
            "poll": "read_receipt",
            "rollback": "reverse_receipt",
        },
        "arguments": {
            "search": {"query": "search_text"},
            "inspect_device": {"device_id": "asset_ref"},
            "inspect_routines": {"device_id": "asset_ref"},
            "apply_change": {
                "action": "command",
                "target_id": "object_ref",
                "payload": "inputs",
            },
            "poll": {"operation_id": "receipt"},
            "rollback": {"operation_id": "receipt"},
        },
        "payload": {"mode": "setting", "time": "when"},
        "distractors": 3,
    },
    "schema-b": {
        "names": {
            "search": "match_resources",
            "inspect_device": "get_resource_snapshot",
            "inspect_routines": "get_linked_rules",
            "apply_change": "dispatch_command",
            "poll": "get_command_status",
            "rollback": "compensate_command",
        },
        "arguments": {
            "search": {"query": "phrase"},
            "inspect_device": {"device_id": "resource_id"},
            "inspect_routines": {"device_id": "resource_id"},
            "apply_change": {
                "action": "operation",
                "target_id": "resource_id",
                "payload": "parameters",
            },
            "poll": {"operation_id": "command_id"},
            "rollback": {"operation_id": "command_id"},
        },
        "payload": {"mode": "desired_mode", "time": "restore_time"},
        "distractors": 3,
    },
}


@dataclass
class SchemaTransformV2:
    profile_id: str
    names: dict[str, str]
    arguments: dict[str, dict[str, str]]
    payload: dict[str, str]
    distractor_count: int
    _runtime_names: dict[str, str] = field(default_factory=dict)

    def public_tools(self, tools: tuple[PublicTool, ...]) -> tuple[PublicTool, ...]:
        if self.profile_id == "identity":
            return tools
        shifted: list[PublicTool] = []
        for tool in tools:
            operation = self._operation(tool.name)
            if operation is None:
                shifted.append(tool)
                continue
            self._runtime_names[operation] = tool.name
            name = self.names.get(operation, tool.name)
            parameters = self._public_parameters(operation, tool.parameters)
            shifted.append(
                PublicTool(
                    name=name,
                    description=(
                        f"Contract profile {self.profile_id}. {tool.description} "
                        "Treat names and fields as authoritative; do not infer a previous schema."
                    ),
                    parameters=parameters,
                )
            )
        shifted.extend(_distractor_tools(self.distractor_count))
        return tuple(shifted)

    def public_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.profile_id == "identity":
            return messages
        shifted = deepcopy(messages)
        for message in shifted:
            if message.get("role") == "tool" and isinstance(message.get("name"), str):
                operation = self._operation(message["name"])
                if operation is not None:
                    message["name"] = self.names.get(operation, message["name"])
            content = message.get("content")
            if message.get("role") == "tool" and isinstance(content, dict):
                runtime_name = content.get("name")
                if isinstance(runtime_name, str):
                    operation = self._operation(runtime_name)
                    if operation is not None:
                        content["name"] = self.names.get(operation, runtime_name)
            for call in message.get("tool_calls", []):
                function = call.get("function", {})
                runtime_name = function.get("name")
                if not isinstance(runtime_name, str):
                    continue
                operation = self._operation(runtime_name)
                if operation is None:
                    continue
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        continue
                if isinstance(arguments, dict):
                    function["arguments"] = self._to_public_arguments(operation, arguments)
                function["name"] = self.names.get(operation, runtime_name)
        return shifted

    def to_runtime_call(self, call: ToolCall) -> ToolCall:
        if self.profile_id == "identity":
            return call
        operation = next(
            (item for item, name in self.names.items() if name == call.name),
            self._operation(call.name),
        )
        if operation is None:
            return call
        reverse = {
            public: canonical
            for canonical, public in self.arguments.get(operation, {}).items()
        }
        arguments = {
            reverse.get(key, key): deepcopy(value) for key, value in call.arguments.items()
        }
        if operation == "apply_change" and isinstance(arguments.get("payload"), dict):
            payload_reverse = {public: canonical for canonical, public in self.payload.items()}
            arguments["payload"] = {
                payload_reverse.get(key, key): value for key, value in arguments["payload"].items()
            }
        runtime_name = self._runtime_names.get(operation, _KNOWN_NAMES[operation][0])
        return ToolCall(runtime_name, arguments)

    def _operation(self, name: str) -> str | None:
        for operation, names in _KNOWN_NAMES.items():
            if name in names or name == self.names.get(operation):
                return operation
        return None

    def _public_parameters(self, operation: str, schema: dict[str, Any]) -> dict[str, Any]:
        parameters = deepcopy(schema)
        mapping = self.arguments.get(operation, {})
        properties = parameters.get("properties", {})
        parameters["properties"] = {
            mapping.get(key, key): value for key, value in properties.items()
        }
        parameters["required"] = [
            mapping.get(key, key) for key in parameters.get("required", [])
        ]
        if operation == "apply_change":
            payload_name = mapping.get("payload", "payload")
            payload = parameters["properties"].get(payload_name, {})
            nested = payload.get("properties", {})
            payload["properties"] = {
                self.payload.get(key, key): value for key, value in nested.items()
            }
        return parameters

    def _to_public_arguments(
        self, operation: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        mapping = self.arguments.get(operation, {})
        public = {
            mapping.get(key, key): deepcopy(value)
            for key, value in arguments.items()
            if key != "idempotency_key"
        }
        payload_name = mapping.get("payload", "payload")
        if operation == "apply_change" and isinstance(public.get(payload_name), dict):
            public[payload_name] = {
                self.payload.get(key, key): value
                for key, value in public[payload_name].items()
            }
        return public


def schema_profile(profile_id: str) -> SchemaTransformV2:
    if profile_id not in _PROFILES:
        raise ValueError(f"unknown V2 schema profile: {profile_id}")
    profile = _PROFILES[profile_id]
    return SchemaTransformV2(
        profile_id=profile_id,
        names=deepcopy(profile["names"]),
        arguments=deepcopy(profile["arguments"]),
        payload=deepcopy(profile["payload"]),
        distractor_count=int(profile["distractors"]),
    )


def schema_shift_functiongemma_record(
    record: dict[str, Any], profile_id: str
) -> dict[str, Any]:
    """Apply one public contract profile to a complete native FunctionGemma record."""
    shifted = deepcopy(record)
    transform = schema_profile(profile_id)
    public_tools = tuple(
        PublicTool(
            name=str(item["function"]["name"]),
            description=str(item["function"].get("description", "")),
            parameters=deepcopy(item["function"]["parameters"]),
        )
        for item in shifted.get("tools", [])
    )
    shifted["tools"] = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(tool.parameters),
            },
        }
        for tool in transform.public_tools(public_tools)
    ]
    shifted["prompt"] = transform.public_messages(shifted["prompt"])
    shifted["completion"] = transform.public_messages(shifted["completion"])
    shifted["metadata"] = dict(shifted.get("metadata", {}))
    shifted["metadata"]["schema_profile"] = profile_id
    return shifted


def _distractor_tools(count: int) -> list[PublicTool]:
    candidates = [
        PublicTool(
            "lookup_asset_docs",
            "Search documentation only; this does not inspect live device state.",
            {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
        ),
        PublicTool(
            "estimate_energy",
            "Estimate energy usage; this never changes a device.",
            {
                "type": "object",
                "properties": {"asset_ref": {"type": "string"}},
                "required": ["asset_ref"],
                "additionalProperties": False,
            },
        ),
        PublicTool(
            "preview_change",
            "Render a human preview only; this does not submit or validate a mutation.",
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        ),
    ]
    return candidates[:count]
