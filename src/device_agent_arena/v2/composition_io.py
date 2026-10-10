from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from device_agent_arena.v1.models import PlanStep
from device_agent_arena.v2.composition_models import (
    V2ActionSpec,
    V2CompositionGoal,
    V2CompositionTask,
    V2UserActionSpec,
)


def composition_task_from_dict(data: dict[str, Any]) -> V2CompositionTask:
    return V2CompositionTask(
        id=str(data["id"]),
        version=str(data["version"]),
        split=data["split"],
        seed=int(data["seed"]),
        slice_id=data["slice_id"],
        domain=data["domain"],
        instruction=str(data["instruction"]),
        public_context=dict(data["public_context"]),
        initial_world=dict(data["initial_world"]),
        safety_policy=tuple(data["safety_policy"]),
        user_actions=tuple(V2UserActionSpec(**item) for item in data["user_actions"]),
        action_specs=tuple(V2ActionSpec(**item) for item in data["action_specs"]),
        goal=V2CompositionGoal(
            state_requirements=dict(data["goal"].get("state_requirements", {})),
            required_milestones=tuple(data["goal"].get("required_milestones", ())),
            precedence=tuple(
                tuple(item) for item in data["goal"].get("precedence", ())
            ),
            forbidden_actions=tuple(data["goal"].get("forbidden_actions", ())),
            allowed_mutation_targets=tuple(
                data["goal"].get("allowed_mutation_targets", ())
            ),
            terminal_behavior=data["goal"].get("terminal_behavior", "answer"),
            max_cost=float(data["goal"].get("max_cost", 20.0)),
        ),
        oracle_plan=tuple(PlanStep(**item) for item in data["oracle_plan"]),
        provenance=dict(data["provenance"]),
    )


def load_composition_tasks(path: str | Path) -> tuple[V2CompositionTask, ...]:
    tasks = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            tasks.append(composition_task_from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid composition task at line {line_number}") from exc
    return tuple(tasks)


def write_composition_tasks(
    path: str | Path, tasks: tuple[V2CompositionTask, ...]
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite composition tasks: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(asdict(task), ensure_ascii=False, sort_keys=True) + "\n"
            for task in tasks
        ),
        encoding="utf-8",
        newline="\n",
    )
