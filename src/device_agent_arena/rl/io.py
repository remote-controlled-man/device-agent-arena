from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from device_agent_arena.domain.models import (
    DeviceState,
    GradeResult,
    TaskSpec,
    TrajectoryEvent,
    TrialResult,
)
from device_agent_arena.rl.credit import CreditAssigner, CreditStrategy
from device_agent_arena.rl.rewards import RewardModel
from device_agent_arena.rl.rollout import RolloutTrajectory

ROLLOUT_FORMAT_VERSION = "rl-rollout-v0.1"


def write_rollouts(
    data_path: str | Path,
    manifest_path: str | Path,
    trajectories: tuple[RolloutTrajectory, ...],
    *,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("at least one rollout trajectory is required")
    if "run_id" not in run_metadata or not str(run_metadata["run_id"]).strip():
        raise ValueError("run_metadata must contain a non-empty run_id")
    gammas = {trajectory.gamma for trajectory in trajectories}
    if len(gammas) != 1:
        raise ValueError("all rollout trajectories must use the same gamma")
    gamma = gammas.pop()

    header = {
        "record_type": "rollout_header",
        "format_version": ROLLOUT_FORMAT_VERSION,
        "gamma": gamma,
        **run_metadata,
    }
    records = [header]
    records.extend(
        {
            "record_type": "trajectory",
            "task_id": trajectory.task_id,
            "generation": trajectory.generation,
            "credit_strategy": trajectory.credit_strategy.value,
            "advantages": list(trajectory.advantages),
            "reward": asdict(trajectory.reward),
            "trial": asdict(trajectory.trial),
        }
        for trajectory in trajectories
    )
    payload = (
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
        + "\n"
    ).encode("utf-8")
    manifest = {
        "format_version": ROLLOUT_FORMAT_VERSION,
        "run_id": str(run_metadata["run_id"]),
        "trajectory_count": len(trajectories),
        "successful_trajectory_count": sum(
            trajectory.trial.grade.success for trajectory in trajectories
        ),
        "gamma": gamma,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    destination = Path(data_path)
    manifest_destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    manifest_destination.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest


def validate_rollouts(
    data_path: str | Path,
    manifest_path: str | Path,
    *,
    tasks: Iterable[TaskSpec],
) -> dict[str, Any]:
    payload = Path(data_path).read_bytes()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    digest = hashlib.sha256(payload).hexdigest()
    if manifest.get("sha256") != digest:
        raise ValueError("rollout artifact SHA-256 does not match manifest")

    records = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    if not records or records[0].get("record_type") != "rollout_header":
        raise ValueError("rollout artifact must start with a rollout_header")
    header = records[0]
    if header.get("format_version") != ROLLOUT_FORMAT_VERSION:
        raise ValueError("unsupported rollout format version")
    gamma = header.get("gamma")
    if not isinstance(gamma, (int, float)) or not 0.0 <= gamma <= 1.0:
        raise ValueError("rollout header must contain gamma in [0, 1]")
    if manifest.get("gamma") != gamma:
        raise ValueError("rollout gamma does not match manifest")

    task_by_id = {task.id: task for task in tasks}
    if len(task_by_id) == 0:
        raise ValueError("at least one task is required to validate rollouts")
    grouped_records: dict[str, list[dict[str, Any]]] = {}
    successful = 0
    for record in records[1:]:
        if record.get("record_type") != "trajectory":
            raise ValueError("unexpected non-trajectory record")
        task_id = record.get("task_id")
        if task_id not in task_by_id:
            raise ValueError(f"rollout references unknown task: {task_id}")
        trial = _trial_from_dict(record["trial"])
        reward = RewardModel().score(task_by_id[task_id], trial)
        if record.get("reward") != _json_value(asdict(reward)):
            raise ValueError(f"rollout reward does not recompute for {task_id}")
        successful += int(trial.grade.success)
        grouped_records.setdefault(task_id, []).append(record)

    for task_id, group in grouped_records.items():
        ordered = sorted(group, key=lambda record: record["generation"])
        strategies = {record.get("credit_strategy") for record in ordered}
        if len(strategies) != 1:
            raise ValueError(f"mixed credit strategies for {task_id}")
        strategy = CreditStrategy(strategies.pop())
        rewards = tuple(
            RewardModel().score(task_by_id[task_id], _trial_from_dict(record["trial"]))
            for record in ordered
        )
        expected = CreditAssigner(gamma=gamma).assign(rewards, strategy=strategy)
        actual = tuple(tuple(record.get("advantages", ())) for record in ordered)
        if actual != expected:
            raise ValueError(
                f"rollout advantages do not recompute for {task_id}: "
                f"expected {expected}, got {actual}"
            )

    trajectory_count = len(records) - 1
    if manifest.get("trajectory_count") != trajectory_count:
        raise ValueError("rollout trajectory count does not match manifest")
    if manifest.get("successful_trajectory_count") != successful:
        raise ValueError("rollout success count does not match manifest")
    return {
        "format_version": ROLLOUT_FORMAT_VERSION,
        "run_id": header["run_id"],
        "trajectory_count": trajectory_count,
        "successful_trajectory_count": successful,
        "valid": True,
    }


def _trial_from_dict(data: dict[str, Any]) -> TrialResult:
    grade = data["grade"]
    return TrialResult(
        task_id=data["task_id"],
        agent_name=data["agent_name"],
        termination=data["termination"],
        final_state=DeviceState(**data["final_state"]),
        events=tuple(TrajectoryEvent(**event) for event in data["events"]),
        grade=GradeResult(
            success=grade["success"],
            checks=grade["checks"],
            diagnostics=grade.get("diagnostics", {}),
        ),
        elapsed_ms=data["elapsed_ms"],
    )


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
