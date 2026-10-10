from __future__ import annotations

import json

from device_agent_arena.domain.models import ToolCall
from device_agent_arena.v1.generate import generate_tasks
from device_agent_arena.v1.sft import messages_for_observation
from device_agent_arena.v2.protocol import v2_public_tools
from device_agent_arena.v2.schema_shift import (
    schema_profile,
    schema_shift_functiongemma_record,
)


def test_schema_shift_changes_names_arguments_payload_and_adds_distractors() -> None:
    observation = generate_tasks("dev", count=1, root_seed=7)[0].public_observation(
        step=0, events=()
    )
    transform = schema_profile("schema-a")

    tools = transform.public_tools(v2_public_tools(observation))
    mutation = next(tool for tool in tools if tool.name == "commit_asset_change")

    assert set(mutation.parameters["properties"]) == {"command", "object_ref", "inputs"}
    assert set(mutation.parameters["required"]) == {"command", "object_ref", "inputs"}
    assert "setting" in mutation.parameters["properties"]["inputs"]["properties"]
    assert "mode" not in mutation.parameters["properties"]["inputs"]["properties"]
    assert len(tools) == len(v2_public_tools(observation)) + 3


def test_schema_shift_round_trips_model_call_to_runtime_contract() -> None:
    transform = schema_profile("schema-a")

    runtime = transform.to_runtime_call(
        ToolCall(
            "commit_asset_change",
            {
                "command": "set_mode",
                "object_ref": "node-1",
                "inputs": {"setting": "focus"},
            },
        )
    )

    assert runtime == ToolCall(
        "apply_change",
        {"action": "set_mode", "target_id": "node-1", "payload": {"mode": "focus"}},
    )


def test_schema_shift_rewrites_visible_history_consistently() -> None:
    transform = schema_profile("schema-a")
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "y"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "apply_change",
                        "arguments": json.dumps(
                            {
                                "action": "schedule_restore",
                                "target_id": "node-1",
                                "payload": {"time": "07:00", "mode": "normal"},
                                "idempotency_key": "runtime-only",
                            }
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "name": "apply_change", "content": "{}"},
    ]

    shifted = transform.public_messages(messages)
    function = shifted[2]["tool_calls"][0]["function"]

    assert function["name"] == "commit_asset_change"
    assert function["arguments"] == {
        "command": "schedule_restore",
        "object_ref": "node-1",
        "inputs": {"when": "07:00", "setting": "normal"},
    }
    assert shifted[3]["name"] == "commit_asset_change"


def test_identity_profile_is_a_true_no_op() -> None:
    observation = generate_tasks("dev", count=1, root_seed=7)[0].public_observation(
        step=0, events=()
    )
    tools = v2_public_tools(observation)
    messages = messages_for_observation(observation)
    transform = schema_profile("identity")

    assert transform.public_tools(tools) == tools
    assert transform.public_messages(messages) == messages
    assert transform.to_runtime_call(ToolCall("search_devices", {"query": "lamp"})) == (
        ToolCall("search_devices", {"query": "lamp"})
    )


def test_schema_shift_rewrites_native_functiongemma_record_consistently() -> None:
    record = {
        "prompt": [
            {"role": "developer", "content": "use tools"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "inspect_device",
                            "arguments": {"device_id": "node-1"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": {
                    "name": "inspect_device",
                    "response": {"ok": True, "data": {"id": "node-1"}},
                },
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "apply_change",
                            "arguments": {
                                "action": "set_mode",
                                "target_id": "node-1",
                                "payload": {"mode": "focus"},
                            },
                        },
                    }
                ],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "inspect_device",
                    "description": "inspect",
                    "parameters": {
                        "type": "object",
                        "properties": {"device_id": {"type": "string"}},
                        "required": ["device_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_change",
                    "description": "mutate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "target_id": {"type": "string"},
                            "payload": {
                                "type": "object",
                                "properties": {"mode": {"type": "string"}},
                            },
                        },
                        "required": ["action", "target_id", "payload"],
                    },
                },
            },
        ],
        "metadata": {"task_id": "train-1"},
    }

    shifted = schema_shift_functiongemma_record(record, "schema-a")

    assert shifted["prompt"][1]["tool_calls"][0]["function"] == {
        "name": "read_asset",
        "arguments": {"asset_ref": "node-1"},
    }
    assert shifted["prompt"][2]["content"]["name"] == "read_asset"
    assert shifted["completion"][0]["tool_calls"][0]["function"] == {
        "name": "commit_asset_change",
        "arguments": {
            "command": "set_mode",
            "object_ref": "node-1",
            "inputs": {"setting": "focus"},
        },
    }
    assert {tool["function"]["name"] for tool in shifted["tools"]} >= {
        "read_asset",
        "commit_asset_change",
        "lookup_asset_docs",
    }
    assert shifted["metadata"]["schema_profile"] == "schema-a"
