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
from device_agent_arena.v1.environment import PartialContractEnvironment
from device_agent_arena.v1.models import (
    DialogueAction,
    V1Observation,
    V1TaskSpec,
    V1TrialResult,
)


class AgentV1(Protocol):
    name: str

    def act(self, observation: V1Observation) -> ToolCall | DialogueAction: ...


_V2_MODEL_RUN_CAPABILITY = object()


class EvaluationHarnessV1:
    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig(max_steps=12)

    def run(self, task: V1TaskSpec, agent: AgentV1) -> V1TrialResult:
        environment = PartialContractEnvironment(self.config)
        observation = environment.reset(task)
        while True:
            action = agent.act(observation)
            drain_telemetry = getattr(agent, "drain_telemetry", None)
            telemetry = drain_telemetry() if callable(drain_telemetry) else None
            transition = environment.step(action, model_event=telemetry)
            if transition.done:
                return environment.finish(agent.name)
            observation = transition.observation


def _task_payload(tasks: tuple[V1TaskSpec, ...]) -> bytes:
    return (
        "".join(
            json.dumps(asdict(task), ensure_ascii=False, sort_keys=True) + "\n"
            for task in tasks
        )
    ).encode("utf-8")


def run_v1_tasks(
    *,
    tasks: Iterable[V1TaskSpec],
    agent_factory: Callable[[V1TaskSpec], AgentV1],
    agent_id: str,
    split: str,
    output: str | Path,
    config: HarnessConfig | None = None,
    run_metadata: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    _v2_model_run_capability: object | None = None,
) -> Path:
    task_list = tuple(tasks)
    if not task_list:
        raise ValueError("v1 run requires at least one task")
    if agent_id.startswith("v2-model:") and (
        _v2_model_run_capability is not _V2_MODEL_RUN_CAPABILITY
    ):
        raise ValueError("V2 model-labelled runs must use the trusted model runner")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing run artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_config = config or HarnessConfig(max_steps=12)
    payload = _task_payload(task_list)
    header = {
        "record_type": "run_header",
        "format_version": "v1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "agent_id": agent_id,
        "split": split,
        "task_count": len(task_list),
        "task_suite_sha256": hashlib.sha256(payload).hexdigest(),
        "task_ids_sha256": hashlib.sha256(
            ("\n".join(task.id for task in task_list) + "\n").encode("utf-8")
        ).hexdigest(),
        "grader_version": "deviceops-ul-v1.0",
        "harness_config": asdict(resolved_config),
        "source": {**git_provenance(Path.cwd()), "tree_sha256": source_tree_sha256()},
        "agent_config": run_metadata or {},
    }
    harness = EvaluationHarnessV1(resolved_config)
    records: list[dict] = [header]
    for index, task in enumerate(task_list, start=1):
        trial = harness.run(task, agent_factory(task))
        record = asdict(trial)
        record.update(
            {
                "record_type": "trial",
                "categories": list(task.categories),
                "required_clarifications": list(task.goal.required_clarifications),
                "required_confirmations": list(task.goal.required_confirmations),
                "fault_planned": bool(task.fault_plan),
            }
        )
        records.append(record)
        if progress_callback is not None:
            progress_callback(index, len(task_list))
    serialized = (
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    ).encode("utf-8")
    _publish_exclusive(destination, serialized)
    return destination


def _run_attested_v2_model_tasks(**kwargs) -> Path:
    """Internal capability used by the V2 concrete-backend runner."""
    return run_v1_tasks(
        **kwargs,
        _v2_model_run_capability=_V2_MODEL_RUN_CAPABILITY,
    )
