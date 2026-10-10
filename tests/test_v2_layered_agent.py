from __future__ import annotations

from dataclasses import dataclass

import pytest

from device_agent_arena.domain.models import ToolCall
from device_agent_arena.harness.types import HarnessConfig
from device_agent_arena.v2.composition_environment import CompositionEnvironment
from device_agent_arena.v2.composition_generate import generate_composition_tasks
from device_agent_arena.v2.composition_layered import (
    LayeredCompositionAgent,
    expected_fast_path_action,
)
from device_agent_arena.v2.composition_runner import (
    CompositionHarness,
    run_attested_layered_composition_model_tasks,
    run_composition_tasks,
)


@dataclass
class _StaticAgent:
    action: ToolCall
    role_hint: str

    @property
    def name(self) -> str:
        return self.role_hint

    def act(self, _observation):
        return self.action

    def drain_telemetry(self):
        return {"input_tokens": 10, "output_tokens": 2, "raw_model_output": "raw"}


def _dual_task():
    return next(
        task
        for task in generate_composition_tasks("train", count=18, root_seed=20260902)
        if task.slice_id == "dual_control"
        and task.provenance.get("variant") == "standard"
    )


def test_fast_path_is_limited_to_publicly_verifiable_states() -> None:
    task = _dual_task()
    environment = CompositionEnvironment()
    observation = environment.reset(task)
    target_id = str(task.public_context["target_id"])
    assert expected_fast_path_action(observation) == ToolCall(
        "inspect_target", {"target_id": target_id}
    )

    for action in (
        ToolCall("inspect_target", {"target_id": target_id}),
        ToolCall("inspect_relations", {"target_id": target_id}),
    ):
        observation = environment.step(action).observation
    assert expected_fast_path_action(observation) is None

    relation = next(
        item for item in task.initial_world["relations"] if item.get("user_action_key")
    )
    observation = environment.step(
        ToolCall(
            "request_user_action",
            {"action_key": relation["user_action_key"], "message": "请完成。"},
        )
    ).observation
    assert expected_fast_path_action(observation) == ToolCall(
        "inspect_target", {"target_id": target_id}
    )


def test_fast_path_does_not_mask_unavailable_user_action_evidence() -> None:
    task = next(
        task
        for task in generate_composition_tasks("train", count=18, root_seed=20260902)
        if task.slice_id == "composed_infeasible"
        and task.provenance.get("variant") == "infeasible"
    )
    environment = CompositionEnvironment()
    environment.reset(task)
    target_id = str(task.public_context["target_id"])
    for action in (
        ToolCall("inspect_target", {"target_id": target_id}),
        ToolCall("inspect_relations", {"target_id": target_id}),
    ):
        environment.step(action)
    relation = next(
        item for item in task.initial_world["relations"] if item.get("user_action_key")
    )
    transition = environment.step(
        ToolCall(
            "request_user_action",
            {"action_key": relation["user_action_key"], "message": "请完成。"},
        )
    )

    assert transition.tool_result is not None
    assert transition.tool_result.code == "user_action_unavailable"
    assert expected_fast_path_action(transition.observation) is None


def test_layered_agent_accepts_exact_router_proposal_without_planner() -> None:
    task = _dual_task()
    observation = CompositionEnvironment().reset(task)
    target_id = str(task.public_context["target_id"])
    router = _StaticAgent(ToolCall("inspect_target", {"target_id": target_id}), "router")
    planner = _StaticAgent(ToolCall("refuse_task", {"reason": "unused"}), "planner")
    agent = LayeredCompositionAgent(router=router, planner=planner)

    action = agent.act(observation)
    telemetry = agent.drain_telemetry()

    assert action == router.action
    assert len(telemetry) == 1
    assert telemetry[0]["model_role"] == "router"
    assert telemetry[0]["decision_used"] is True


def test_layered_agent_rejects_router_deviation_and_falls_back() -> None:
    task = _dual_task()
    observation = CompositionEnvironment().reset(task)
    target_id = str(task.public_context["target_id"])
    router = _StaticAgent(ToolCall("poll_action", {"operation_id": "fake"}), "router")
    planner = _StaticAgent(
        ToolCall("inspect_target", {"target_id": target_id}), "planner"
    )
    agent = LayeredCompositionAgent(router=router, planner=planner)

    action = agent.act(observation)
    telemetry = agent.drain_telemetry()

    assert action == planner.action
    assert [item["model_role"] for item in telemetry] == ["router", "planner"]
    assert telemetry[0]["decision_used"] is False
    assert telemetry[0]["rejection_reason"] == "fast_path_contract_mismatch"
    assert telemetry[1]["decision_used"] is True


def test_harness_records_rejected_router_and_used_planner_costs() -> None:
    task = _dual_task()
    target_id = str(task.public_context["target_id"])
    agent = LayeredCompositionAgent(
        router=_StaticAgent(
            ToolCall("poll_action", {"operation_id": "fake"}), "router"
        ),
        planner=_StaticAgent(
            ToolCall("inspect_target", {"target_id": target_id}), "planner"
        ),
    )

    trial = CompositionHarness(HarnessConfig(max_steps=1)).run(task, agent)
    model_events = [event for event in trial.events if event.kind == "model"]

    assert [event.payload["model_role"] for event in model_events] == [
        "router",
        "planner",
    ]
    assert [event.payload["decision_used"] for event in model_events] == [False, True]


def test_generic_runner_cannot_masquerade_as_layered_model(tmp_path) -> None:
    task = _dual_task()
    with pytest.raises(ValueError, match="trusted model runner"):
        run_composition_tasks(
            tasks=[task],
            agent_factory=lambda _task: _StaticAgent(
                ToolCall("finish_task", {"message": "fake"}), "fake"
            ),
            agent_id="v2-composition-layered:fake",
            split="train",
            output=tmp_path / "fake.jsonl",
        )


def test_layered_runner_requires_two_concrete_transformers_backends(tmp_path) -> None:
    with pytest.raises(TypeError, match="two concrete TransformersBackend"):
        run_attested_layered_composition_model_tasks(
            tasks=[_dual_task()],
            router_backend=object(),
            planner_backend=object(),
            split="train",
            output=tmp_path / "fake.jsonl",
        )
