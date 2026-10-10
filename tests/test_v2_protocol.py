from __future__ import annotations

import json

from device_agent_arena.agents.model_agent import ModelResponse
from device_agent_arena.domain.models import ToolCall, TrajectoryEvent
from device_agent_arena.v1.generate import generate_tasks
from device_agent_arena.v1.models import DialogueAction
from device_agent_arena.v2.composition_generate import generate_composition_tasks
from device_agent_arena.v2.protocol import (
    V2ModelAgent,
    functiongemma_record,
    v2_public_tools,
)
from device_agent_arena.v2.schema_shift import schema_profile


class FakeBackend:
    name = "fake-functiongemma"
    is_functiongemma = True

    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.messages: list[dict] | None = None
        self.tools: list[dict] | None = None

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        self.messages = messages
        self.tools = tools
        return self.response


def test_v2_contract_exposes_payload_and_idempotency_semantics() -> None:
    observation = generate_tasks("dev", count=1, root_seed=7)[0].public_observation(
        step=0,
        events=(),
    )

    tools = v2_public_tools(observation)
    mutation = next(tool for tool in tools if "action" in tool.parameters["properties"])
    payload = mutation.parameters["properties"]["payload"]

    assert "time" in payload["properties"]
    assert "restore_at" not in payload["properties"]
    assert "idempotency_key" not in mutation.parameters["properties"]
    assert "idempotency_key" not in mutation.parameters["required"]
    assert {
        "request_clarification",
        "request_confirmation",
        "finish_task",
        "refuse_task",
    } <= {tool.name for tool in tools}


def test_v2_public_tools_do_not_duplicate_composition_interactions() -> None:
    observation = generate_composition_tasks(
        "dev", count=9, root_seed=20260902
    )[0].public_observation(step=0, events=())

    tools = v2_public_tools(observation)
    names = [tool.name for tool in tools]

    assert len(names) == len(set(names))
    assert "request_user_action" in names
    assert names.count("finish_task") == 1
    assert names.count("refuse_task") == 1


def test_v2_model_agent_maps_native_interaction_call_to_dialogue_action() -> None:
    backend = FakeBackend(
        ModelResponse(
            text="",
            tool_calls=(
                ToolCall(
                    "request_clarification",
                    {"field": "device", "question": "你指的是哪一盏灯？"},
                ),
            ),
        )
    )
    observation = generate_tasks("dev", count=1, root_seed=7)[0].public_observation(
        step=0,
        events=(),
    )

    action = V2ModelAgent(backend).act(observation)

    assert action == DialogueAction("clarify", "你指的是哪一盏灯？", field="device")
    assert backend.tools is not None
    assert any(
        item["function"]["name"] == "request_clarification"
        for item in backend.tools
    )


def test_v2_composition_mode_uses_only_environment_native_tools() -> None:
    backend = FakeBackend(
        ModelResponse(
            text="",
            tool_calls=(ToolCall("finish_task", {"message": "完成"}),),
        )
    )
    observation = generate_composition_tasks(
        "train", count=9, root_seed=20260902
    )[0].public_observation(step=1, events=())

    action = V2ModelAgent(backend, interaction_mode="tool").act(observation)

    assert action == ToolCall("finish_task", {"message": "完成"})
    assert backend.tools is not None
    assert {item["function"]["name"] for item in backend.tools} == {
        tool.name for tool in observation.tools
    }
    assert "request_confirmation" not in {
        item["function"]["name"] for item in backend.tools
    }


def test_v2_model_agent_name_identifies_schema_profile() -> None:
    backend = FakeBackend(ModelResponse(text="done"))

    assert V2ModelAgent(backend).name == "v2-model:fake-functiongemma"
    assert V2ModelAgent(
        backend, contract_transform=schema_profile("schema-a")
    ).name == "v2-model:fake-functiongemma:schema-a"


def test_v2_functiongemma_runtime_history_matches_native_training_protocol() -> None:
    backend = FakeBackend(
        ModelResponse(
            text="",
            tool_calls=(ToolCall("finish_task", {"message": "完成"}),),
        )
    )
    task = generate_tasks("dev", count=1, root_seed=7)[0]
    observation = task.public_observation(
        step=2,
        events=(
            TrajectoryEvent(
                step=1,
                kind="message",
                payload={
                    "kind": "clarify",
                    "content": "哪一台？",
                    "field": "device",
                    "operation": None,
                },
            ),
            TrajectoryEvent(
                step=1,
                kind="user",
                payload={"field": "device", "content": "主灯"},
            ),
        ),
    )

    V2ModelAgent(backend).act(observation)

    assert backend.messages is not None
    interaction = backend.messages[2]["tool_calls"][0]["function"]
    assert interaction["name"] == "request_clarification"
    assert backend.messages[3]["role"] == "tool"
    assert backend.messages[3]["name"] == "request_clarification"


def test_functiongemma_record_uses_native_calls_and_tool_responses() -> None:
    record = {
        "prompt": [
            {"role": "system", "content": "安全执行"},
            {"role": "user", "content": "把卧室灯调暗"},
            {"role": "assistant", "content": "[CLARIFY field=device] 哪一盏？"},
            {"role": "user", "content": json.dumps({"field": "device", "content": "主灯"})},
        ],
        "completion": [
            {"role": "assistant", "content": "[CONFIRM operation=set_mode:main] 是否继续？"}
        ],
        "tools": [],
        "metadata": {"task_id": "x", "decision_kind": "confirm"},
    }

    converted = functiongemma_record(record)

    assert converted["prompt"][0]["role"] == "developer"
    clarify = converted["prompt"][2]["tool_calls"][0]["function"]
    assert clarify["name"] == "request_clarification"
    assert clarify["arguments"]["field"] == "device"
    assert converted["prompt"][3]["role"] == "tool"
    assert converted["prompt"][3]["content"]["name"] == "request_clarification"
    confirm = converted["completion"][0]["tool_calls"][0]["function"]
    assert confirm["name"] == "request_confirmation"
    assert converted["metadata"]["protocol"] == "functiongemma-native-v2"


def test_functiongemma_record_removes_runtime_managed_idempotency_keys() -> None:
    record = {
        "prompt": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        "completion": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "submit_mutation",
                            "arguments": json.dumps(
                                {
                                    "action": "set_mode",
                                    "target_id": "node-1",
                                    "payload": {"mode": "focus"},
                                    "idempotency_key": "model-should-not-own-this",
                                }
                            ),
                        },
                    }
                ],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "submit_mutation",
                    "description": "mutate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "target_id": {"type": "string"},
                            "payload": {"type": "object"},
                            "idempotency_key": {"type": "string"},
                        },
                        "required": ["action", "target_id", "payload", "idempotency_key"],
                    },
                },
            }
        ],
        "metadata": {},
    }

    converted = functiongemma_record(record)
    arguments = converted["completion"][0]["tool_calls"][0]["function"]["arguments"]
    mutation_schema = converted["tools"][0]["function"]["parameters"]

    assert "idempotency_key" not in arguments
    assert "idempotency_key" not in mutation_schema["properties"]
    assert "idempotency_key" not in mutation_schema["required"]
