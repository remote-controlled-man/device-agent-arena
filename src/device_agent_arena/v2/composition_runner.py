from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from device_agent_arena.artifacts.run_io import (
    _publish_exclusive,
    git_provenance,
    source_tree_sha256,
)
from device_agent_arena.domain.models import ToolCall
from device_agent_arena.harness.types import HarnessConfig
from device_agent_arena.v1.models import DialogueAction, V1Observation, V1TrialResult
from device_agent_arena.v2.composition_environment import CompositionEnvironment
from device_agent_arena.v2.composition_models import V2CompositionTask


class CompositionAgent(Protocol):
    name: str

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction: ...


_MODEL_RUN_CAPABILITY = object()


class CompositionHarness:
    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig(max_steps=20)

    def run(
        self, task: V2CompositionTask, agent: CompositionAgent
    ) -> V1TrialResult:
        environment = CompositionEnvironment(self.config)
        observation = environment.reset(task)
        while True:
            action = agent.act(observation)
            drain_telemetry = getattr(agent, "drain_telemetry", None)
            telemetry = drain_telemetry() if callable(drain_telemetry) else None
            transition = environment.step(action, model_event=telemetry)
            if transition.done:
                return environment.finish(agent.name)
            observation = transition.observation


def run_composition_tasks(
    *,
    tasks: Iterable[V2CompositionTask],
    agent_factory: Callable[[V2CompositionTask], CompositionAgent],
    agent_id: str,
    split: str,
    output: str | Path,
    config: HarnessConfig | None = None,
    run_metadata: dict | None = None,
    _model_run_capability: object | None = None,
) -> Path:
    task_list = tuple(tasks)
    if not task_list:
        raise ValueError("composition run requires at least one task")
    if agent_id.startswith(("v2-composition-model:", "v2-composition-layered:")) and (
        _model_run_capability is not _MODEL_RUN_CAPABILITY
    ):
        raise ValueError(
            "model-labelled composition runs must use the trusted model runner"
        )
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite composition run: {destination}")
    resolved_config = config or HarnessConfig(max_steps=20)
    payload = (
        "".join(
            json.dumps(asdict(task), ensure_ascii=False, sort_keys=True) + "\n"
            for task in task_list
        )
    ).encode("utf-8")
    header = {
        "record_type": "run_header",
        "format_version": "deviceops-ul-v2-composition-run-v0.1",
        "created_at": datetime.now(UTC).isoformat(),
        "agent_id": agent_id,
        "split": split,
        "task_count": len(task_list),
        "task_suite_sha256": hashlib.sha256(payload).hexdigest(),
        "task_ids_sha256": hashlib.sha256(
            ("\n".join(task.id for task in task_list) + "\n").encode("utf-8")
        ).hexdigest(),
        "grader_version": "deviceops-ul-v2-composition-v0.1",
        "harness_config": asdict(resolved_config),
        "source": {**git_provenance(Path.cwd()), "tree_sha256": source_tree_sha256()},
        "agent_config": run_metadata or {},
    }
    harness = CompositionHarness(resolved_config)
    records: list[dict] = [header]
    for task in task_list:
        trial = harness.run(task, agent_factory(task))
        record = asdict(trial)
        record.update(
            {
                "record_type": "trial",
                "slice_id": task.slice_id,
                "domain": task.domain,
                "variant": task.provenance.get("variant", "unknown"),
                "scenario_group": task.provenance.get("scenario_group"),
                "topology": task.initial_world["metadata"]["topology"],
            }
        )
        records.append(record)
    serialized = (
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        )
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _publish_exclusive(destination, serialized)
    return destination


def run_attested_composition_model_tasks(
    *,
    tasks: Iterable[V2CompositionTask],
    backend: object,
    split: str,
    output: str | Path,
    config: HarnessConfig | None = None,
) -> Path:
    """Run a model-labelled evaluation through the concrete local HF backend."""
    from device_agent_arena.agents.hf_backend import TransformersBackend
    from device_agent_arena.v2.protocol import V2ModelAgent

    if type(backend) is not TransformersBackend:
        raise TypeError("attested model runs require the concrete TransformersBackend")
    provenance = backend.provenance()
    if not provenance.get("model_artifact_sha256"):
        raise ValueError("attested model runs require a local checkpoint-tree hash")
    return run_composition_tasks(
        tasks=tasks,
        agent_factory=lambda _task: V2ModelAgent(backend, interaction_mode="tool"),
        agent_id=f"v2-composition-model:{backend.name}",
        split=split,
        output=output,
        config=config,
        run_metadata={
            **provenance,
            "attestation_version": "deviceops-ul-model-run-attestation-v2",
            "execution_authority": "deviceops-ul-transformers-runner-v1",
            "protocol": "deviceops-ul-v2-composition-tool-native-v1",
            "candidate_gate_count": 18,
        },
        _model_run_capability=_MODEL_RUN_CAPABILITY,
    )


def run_attested_layered_composition_model_tasks(
    *,
    tasks: Iterable[V2CompositionTask],
    router_backend: object,
    planner_backend: object,
    split: str,
    output: str | Path,
    config: HarnessConfig | None = None,
) -> Path:
    """Run an attested speculative-router + planner composition condition."""
    from device_agent_arena.agents.hf_backend import TransformersBackend
    from device_agent_arena.v2.composition_layered import LayeredCompositionAgent
    from device_agent_arena.v2.protocol import V2ModelAgent

    if (
        type(router_backend) is not TransformersBackend
        or type(planner_backend) is not TransformersBackend
    ):
        raise TypeError("layered runs require two concrete TransformersBackend instances")
    router_provenance = router_backend.provenance()
    planner_provenance = planner_backend.provenance()
    for role, provenance in (
        ("router", router_provenance),
        ("planner", planner_provenance),
    ):
        if not provenance.get("model_artifact_sha256"):
            raise ValueError(f"attested layered {role} requires a local model-tree hash")

    def factory(_task: V2CompositionTask) -> LayeredCompositionAgent:
        return LayeredCompositionAgent(
            router=V2ModelAgent(router_backend, interaction_mode="tool"),
            planner=V2ModelAgent(planner_backend, interaction_mode="tool"),
        )

    return run_composition_tasks(
        tasks=tasks,
        agent_factory=factory,
        agent_id=(
            f"v2-composition-layered:{router_backend.name}=>{planner_backend.name}"
        ),
        split=split,
        output=output,
        config=config,
        run_metadata={
            "attestation_version": "deviceops-ul-layered-model-run-attestation-v1",
            "execution_authority": "deviceops-ul-transformers-runner-v1",
            "protocol": "deviceops-ul-v2-composition-tool-native-v1",
            "fast_path_contract": "deviceops-ul-v2-speculative-fast-path-v1",
            "layers": {
                "router": router_provenance,
                "planner": planner_provenance,
            },
            "candidate_gate_count": 18,
        },
        _model_run_capability=_MODEL_RUN_CAPABILITY,
    )
