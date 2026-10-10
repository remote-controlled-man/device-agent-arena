from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from device_agent_arena.v1.models import (
    PlanStep,
    V1DynamicSpec,
    V1FaultSpec,
    V1Goal,
    V1SafetyPolicy,
    V1TaskSpec,
)

_TASK_FIELDS = {
    "id",
    "version",
    "split",
    "categories",
    "seed",
    "instruction",
    "public_context",
    "tool_profile",
    "initial_world",
    "user_script",
    "safety_policy",
    "fault_plan",
    "dynamic_plan",
    "goal",
    "oracle_plan",
    "provenance",
}


def task_from_dict(data: dict[str, Any]) -> V1TaskSpec:
    unknown = set(data) - _TASK_FIELDS
    missing = _TASK_FIELDS - set(data)
    if unknown:
        raise ValueError(f"unknown task fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing task fields: {sorted(missing)}")
    if data["split"] not in {"train", "dev", "test"}:
        raise ValueError("invalid v1 split")
    safety = data["safety_policy"]
    goal = data["goal"]
    return V1TaskSpec(
        id=str(data["id"]),
        version=str(data["version"]),
        split=data["split"],
        categories=tuple(str(item) for item in data["categories"]),
        seed=int(data["seed"]),
        instruction=str(data["instruction"]),
        public_context=dict(data["public_context"]),
        tool_profile=str(data["tool_profile"]),
        initial_world=dict(data["initial_world"]),
        user_script={str(key): str(value) for key, value in data["user_script"].items()},
        safety_policy=V1SafetyPolicy(
            high_impact_actions=tuple(safety["high_impact_actions"]),
            shared_change_requires_confirmation=bool(
                safety["shared_change_requires_confirmation"]
            ),
            policy_text=tuple(safety["policy_text"]),
        ),
        fault_plan=tuple(V1FaultSpec(**item) for item in data["fault_plan"]),
        dynamic_plan=tuple(V1DynamicSpec(**item) for item in data["dynamic_plan"]),
        goal=V1Goal(
            state_requirements=dict(goal["state_requirements"]),
            required_tools=tuple(goal["required_tools"]),
            required_clarifications=tuple(goal["required_clarifications"]),
            required_confirmations=tuple(goal["required_confirmations"]),
            allowed_mutation_targets=tuple(goal["allowed_mutation_targets"]),
            forbidden_mutations=tuple(goal["forbidden_mutations"]),
            terminal_behavior=goal["terminal_behavior"],
            max_cost=float(goal["max_cost"]),
        ),
        oracle_plan=tuple(
            PlanStep(
                kind=item["kind"],
                name=item.get("name", ""),
                arguments=dict(item.get("arguments", {})),
                field=item.get("field"),
                operation=item.get("operation"),
                content=item.get("content", ""),
            )
            for item in data["oracle_plan"]
        ),
        provenance={str(key): str(value) for key, value in data["provenance"].items()},
    )


def load_tasks(path: str | Path) -> tuple[V1TaskSpec, ...]:
    source = Path(path)
    tasks = tuple(
        task_from_dict(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not tasks:
        raise ValueError(f"{source}: no v1 tasks found")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("duplicate v1 task id")
    return tasks


def write_tasks(path: str | Path, tasks: tuple[V1TaskSpec, ...]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(asdict(task), ensure_ascii=False, sort_keys=True) + "\n" for task in tasks
    )
    destination.write_text(payload, encoding="utf-8", newline="\n")
