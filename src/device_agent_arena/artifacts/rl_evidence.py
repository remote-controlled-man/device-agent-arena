from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from device_agent_arena.artifacts.run_io import load_run, task_suite_payload
from device_agent_arena.dataset.rl_suite import load_rl_task_selection
from device_agent_arena.domain.models import (
    DeviceState,
    TrajectoryEvent,
    TrialResult,
)
from device_agent_arena.domain.task_io import load_tasks
from device_agent_arena.grading.task_grader import grade_task
from device_agent_arena.reporting.aggregate import build_report
from device_agent_arena.reporting.compare import compare_reports
from device_agent_arena.rl.rewards import RewardModel
from device_agent_arena.training.grpo import (
    choose_group_resampling_attempt,
    grpo_config_sha256,
    load_grpo_training_config,
    rank_group_resample_candidates,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _resolve_project_file(project_root: Path, declared_path: str) -> Path:
    candidate = Path(declared_path)
    resolved = (
        (project_root / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"artifact path escapes project root: {declared_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact file is missing: {declared_path}")
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_update_logs(
    trainer_state: dict[str, Any], optimizer_updates: int
) -> list[dict[str, Any]]:
    update_logs = [
        row
        for row in trainer_state.get("log_history", [])
        if isinstance(row, dict)
        and "reward_std" in row
        and "grad_norm" in row
        and isinstance(row.get("step"), int)
    ]
    _require(
        len(update_logs) == optimizer_updates,
        "trainer_state must contain one reward/gradient log for every optimizer update",
    )
    _require(
        {row["step"] for row in update_logs} == set(range(1, optimizer_updates + 1)),
        "trainer_state optimizer-step logs are incomplete or duplicated",
    )
    metric_keys = ("loss", "reward", "reward_std", "grad_norm", "entropy")
    for row in update_logs:
        for key in metric_keys:
            value = row.get(key)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"trainer_state contains non-finite {key} at step {row['step']}")
    return update_logs


def _verify_resampling_task_order(
    *,
    selected_task_ids: list[str],
    actual_task_ids: list[str] | None,
    seed: int,
    update_index: int,
    current_task_id: str,
    attempt_count: int,
) -> list[str]:
    """Recompute the deterministic, without-replacement candidate order."""
    expected = [
        current_task_id,
        *rank_group_resample_candidates(
            task_ids=selected_task_ids,
            current_task_id=current_task_id,
            seed=seed,
            update_index=update_index,
        ),
    ][:attempt_count]
    if actual_task_ids is None:
        return expected
    _require(
        len(actual_task_ids) == len(set(actual_task_ids)),
        "resampling attempts repeat a task within one update",
    )
    _require(
        actual_task_ids == expected,
        "resampling attempts violate deterministic candidate order",
    )
    return expected


def verify_run_grades(
    *,
    task_file_path: str | Path,
    run_path: str | Path,
    task_selection_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind a raw run to frozen tasks and recompute every stored grade."""
    tasks = load_tasks(task_file_path)
    task_by_id = {task.id: task for task in tasks}
    _require(len(task_by_id) == len(tasks), "task file contains duplicate task IDs")
    header, trials = load_run(run_path)
    task_ids = [str(trial.get("task_id")) for trial in trials]
    _require(len(task_ids) == len(set(task_ids)), "raw run contains duplicate task IDs")
    task_selection_sha256 = None
    if task_selection_path is not None:
        split = str(header.get("split"))
        _require(split in {"dev", "test"}, "frozen task selections require a dev/test run")
        frozen_selection = load_rl_task_selection(
            selection_path=task_selection_path,
            source_task_path=task_file_path,
            tasks=tasks,
            split=split,  # type: ignore[arg-type]
        )
        frozen_ids = [task.id for task in frozen_selection]
        _require(
            task_ids == frozen_ids,
            "raw run task order does not match the frozen task selection",
        )
        task_selection_sha256 = _sha256(Path(task_selection_path))
    unknown = [task_id for task_id in task_ids if task_id not in task_by_id]
    _require(not unknown, f"raw run contains task IDs absent from the frozen task file: {unknown}")
    selected_tasks = [task_by_id[task_id] for task_id in task_ids]
    expected_suite_sha256 = hashlib.sha256(task_suite_payload(selected_tasks)).hexdigest()
    expected_ids_sha256 = hashlib.sha256(
        ("\n".join(task_ids) + "\n").encode("utf-8")
    ).hexdigest()
    _require(
        header.get("task_suite_sha256") == expected_suite_sha256,
        "raw run task-suite SHA-256 does not match frozen tasks",
    )
    _require(
        header.get("task_ids_sha256") == expected_ids_sha256,
        "raw run task-ID SHA-256 does not match frozen task order",
    )

    for task, trial in zip(selected_tasks, trials, strict=True):
        _require(
            trial.get("categories") == list(task.categories),
            f"stored categories do not match frozen task: {task.id}",
        )
        _require(
            trial.get("seed") == task.seed,
            f"stored seed does not match frozen task: {task.id}",
        )
        final_state = DeviceState(**trial["final_state"])
        events = tuple(TrajectoryEvent(**event) for event in trial["events"])
        recomputed = asdict(grade_task(goal=task.goal, final_state=final_state, events=events))
        stored = trial.get("grade") or {}
        if (
            stored.get("success") != recomputed["success"]
            or stored.get("checks") != recomputed["checks"]
        ):
            raise ValueError(f"stored grade does not match recomputed grade: {task.id}")
    source = header.get("source") or {}
    return {
        "task_count": len(trials),
        "task_suite_sha256": expected_suite_sha256,
        "task_ids_sha256": expected_ids_sha256,
        "task_selection_sha256": task_selection_sha256,
        "source": source,
    }


def verify_grpo_training_evidence(
    *,
    project_root: str | Path,
    training_result_path: str | Path,
    training_config_path: str | Path | None = None,
    minimum_nonzero_reward_std_rate: float = 0.9,
    minimum_nonzero_gradient_rate: float = 0.9,
) -> dict[str, Any]:
    """Verify a completed GRPO training smoke without requiring model evaluation."""
    if not 0 <= minimum_nonzero_reward_std_rate <= 1:
        raise ValueError("minimum_nonzero_reward_std_rate must be in [0, 1]")
    if not 0 <= minimum_nonzero_gradient_rate <= 1:
        raise ValueError("minimum_nonzero_gradient_rate must be in [0, 1]")

    root = Path(project_root).resolve()
    result_path = _resolve_project_file(root, str(training_result_path))
    result = _load_json(result_path)
    _require(result.get("status") == "completed", "GRPO training result is not completed")
    optimizer_updates = int(result.get("optimizer_updates", 0))
    _require(optimizer_updates > 0, "GRPO result has no completed optimizer updates")
    config = None
    selected_tasks = None
    config_path = None
    if training_config_path is not None:
        config_path = _resolve_project_file(root, str(training_config_path))
        config = load_grpo_training_config(config_path)
        _require(
            grpo_config_sha256(config) == result.get("config_sha256"),
            "config SHA-256 mismatch",
        )
        _require(config.max_updates == optimizer_updates, "config optimizer budget mismatch")
        _require(config.model_id == result.get("model_id"), "config model ID mismatch")
        _require(config.model_revision == result.get("model_revision"), "config revision mismatch")
        _require(config.output_dir == result.get("output_dir"), "config output path mismatch")

        dataset = result.get("dataset") or {}
        data_path = _resolve_project_file(root, str(dataset.get("path", "")))
        manifest_path = _resolve_project_file(root, str(dataset.get("manifest", "")))
        _require(config.train_data == dataset.get("path"), "config training-data path mismatch")
        _require(config.train_manifest == dataset.get("manifest"), "config manifest path mismatch")
        _require(_sha256(data_path) == dataset.get("sha256"), "training data SHA-256 mismatch")
        manifest = _load_json(manifest_path)
        manifest_file = (manifest.get("files") or {}).get(data_path.name) or {}
        _require(manifest_file.get("sha256") == dataset.get("sha256"), "manifest data SHA mismatch")
        _require(
            manifest_file.get("record_count") == dataset.get("record_count"),
            "manifest record count mismatch",
        )
        tasks = load_tasks(data_path)
        selection = dataset.get("selection") or {}
        selection_path = _resolve_project_file(root, str(selection.get("path", "")))
        _require(config.train_selection == selection.get("path"), "config selection path mismatch")
        _require(_sha256(selection_path) == selection.get("sha256"), "selection SHA mismatch")
        selected_tasks = load_rl_task_selection(
            selection_path=selection_path,
            source_task_path=data_path,
            tasks=tasks,
            split="train",
        )
        selected_ids = [task.id for task in selected_tasks]
        _require(selected_ids == selection.get("task_ids"), "selection task order mismatch")
        _require(len(selected_ids) == config.train_task_limit, "selection size mismatch")

    adapter = result.get("adapter") or {}
    adapter_path = _resolve_project_file(root, str(adapter.get("path", "")))
    _require(adapter_path.stat().st_size == adapter.get("size_bytes"), "adapter size mismatch")
    _require(_sha256(adapter_path) == adapter.get("sha256"), "adapter SHA-256 mismatch")

    trainer_state_path = _resolve_project_file(root, str(result.get("trainer_state", "")))
    trainer_state = _load_json(trainer_state_path)
    _require(
        trainer_state.get("global_step") == optimizer_updates,
        "trainer_state global_step does not match the training result",
    )
    _require(
        trainer_state.get("max_steps") == optimizer_updates,
        "training stopped before or after the declared optimizer budget",
    )
    update_logs = _finite_update_logs(trainer_state, optimizer_updates)
    nonzero_reward_std_count = sum(float(row["reward_std"]) > 0 for row in update_logs)
    nonzero_gradient_count = sum(float(row["grad_norm"]) > 0 for row in update_logs)
    nonzero_reward_std_rate = round(nonzero_reward_std_count / optimizer_updates, 6)
    nonzero_gradient_rate = round(nonzero_gradient_count / optimizer_updates, 6)
    clipped_ratios = [
        float(row["completions/clipped_ratio"])
        for row in update_logs
        if row.get("completions/clipped_ratio") is not None
    ]
    mean_token_no_eos_ratio = (
        round(sum(clipped_ratios) / len(clipped_ratios), 6) if clipped_ratios else None
    )

    rollout = result.get("rollout_audit") or {}
    rollout_path = _resolve_project_file(root, str(rollout.get("path", "")))
    _require(_sha256(rollout_path) == rollout.get("sha256"), "rollout audit SHA-256 mismatch")
    rows = [json.loads(line) for line in rollout_path.read_text(encoding="utf-8").splitlines()]
    _require(bool(rows), "rollout audit is empty")
    header, records = rows[0], rows[1:]
    resampling_summary = None
    _require(
        header.get("format_version") == "rl-rollout-audit-v0.1",
        "unsupported rollout audit format",
    )
    _require(header.get("record_count") == len(records), "rollout audit count mismatch")
    _require(rollout.get("record_count") == len(records), "result rollout count mismatch")
    if config is not None and selected_tasks is not None:
        dataset = result["dataset"]
        _require(
            header.get("config_sha256") == result.get("config_sha256"),
            "audit config hash mismatch",
        )
        _require(header.get("dataset_sha256") == dataset.get("sha256"), "audit data hash mismatch")
        _require(
            header.get("selection_sha256") == dataset["selection"].get("sha256"),
            "audit selection hash mismatch",
        )
        if config.max_group_attempts == 1:
            _require(
                len(records) == optimizer_updates * config.num_generations,
                "rollout count does not match optimizer/generation budget",
            )
        else:
            _require(
                optimizer_updates * config.num_generations
                <= len(records)
                <= optimizer_updates * config.num_generations * config.max_group_attempts,
                "rollout count exceeds bounded resampling budget",
            )
            accepted_by_update: dict[int, list[dict[str, Any]]] = {}
            attempts_by_update: dict[int, dict[int, list[dict[str, Any]]]] = {}
            for record in records:
                resampling = record.get("group_resampling") or {}
                update_index = int(resampling.get("update_index", 0))
                _require(1 <= update_index <= optimizer_updates, "invalid resampling update index")
                _require(
                    1 <= int(resampling.get("attempt", 0)) <= config.max_group_attempts,
                    "invalid resampling attempt index",
                )
                attempt_index = int(resampling["attempt"])
                attempts_by_update.setdefault(update_index, {}).setdefault(
                    attempt_index, []
                ).append(record)
                if resampling.get("accepted") is True:
                    accepted_by_update.setdefault(update_index, []).append(record)
            _require(
                set(accepted_by_update) == set(range(1, optimizer_updates + 1)),
                "each update must have an accepted resampling group",
            )
            _require(
                all(len(group) == config.num_generations for group in accepted_by_update.values()),
                "accepted resampling groups must contain num_generations rollouts",
            )
            accepted_signal_counts: Counter[str] = Counter()
            for update_index in range(1, optimizer_updates + 1):
                attempts = attempts_by_update.get(update_index) or {}
                _require(
                    sorted(attempts) == list(range(1, len(attempts) + 1)),
                    "resampling attempts must be contiguous from one",
                )
                _require(len(attempts) <= config.max_group_attempts, "too many attempts")
                signal_types = []
                accepted_attempts = []
                attempt_task_ids: list[str] = []
                for attempt_index, group in sorted(attempts.items()):
                    _require(
                        len(group) == config.num_generations,
                        "resampling attempt must contain num_generations rollouts",
                    )
                    metadata = [record["group_resampling"] for record in group]
                    _require(
                        all(item == metadata[0] for item in metadata),
                        "resampling metadata differs within a group",
                    )
                    _require(
                        all(
                            str(record["task_id"]) == str(metadata[0]["task_id"])
                            for record in group
                        ),
                        "resampling task metadata does not match rollout task",
                    )
                    attempt_task_ids.append(str(metadata[0]["task_id"]))
                    rewards = [float(record["reward_trace"]["total"]) for record in group]
                    successes = sum(bool(record["trial"]["grade"]["success"]) for record in group)
                    reward_std = statistics.pstdev(rewards)
                    signal_type = (
                        "mixed_terminal"
                        if 0 < successes < config.num_generations
                        else (
                            "shaping"
                            if reward_std >= config.minimum_shaping_reward_std
                            else "zero_variance"
                        )
                    )
                    _require(metadata[0]["success_count"] == successes, "success count mismatch")
                    _require(
                        math.isclose(float(metadata[0]["reward_std"]), reward_std, abs_tol=1e-6),
                        "resampling reward std mismatch",
                    )
                    _require(metadata[0]["signal_type"] == signal_type, "signal type mismatch")
                    signal_types.append(signal_type)
                    if metadata[0]["accepted"] is True:
                        accepted_attempts.append(attempt_index)
                _verify_resampling_task_order(
                    selected_task_ids=selected_ids,
                    actual_task_ids=attempt_task_ids,
                    seed=config.seed,
                    update_index=update_index,
                    current_task_id=attempt_task_ids[0],
                    attempt_count=len(attempt_task_ids),
                )
                if "mixed_terminal" in signal_types:
                    _require(
                        signal_types.index("mixed_terminal") == len(signal_types) - 1,
                        "resampling continued after finding a mixed-terminal group",
                    )
                else:
                    _require(
                        len(attempts) == config.max_group_attempts,
                        "resampling stopped before its bound without a mixed group",
                    )
                expected_attempt = choose_group_resampling_attempt(signal_types) + 1
                _require(
                    accepted_attempts == [expected_attempt],
                    "accepted group violates mixed-first selection",
                )
                accepted_signal_type = signal_types[expected_attempt - 1]
                accepted_signal_counts[accepted_signal_type] += 1
                accepted_group = attempts[expected_attempt]
                accepted_rewards = [
                    float(record["reward_trace"]["total"]) for record in accepted_group
                ]
                accepted_reward_mean = statistics.fmean(accepted_rewards)
                accepted_reward_std = statistics.pstdev(accepted_rewards)
                update_log = next(
                    row for row in update_logs if int(row["step"]) == update_index
                )
                _require(
                    math.isclose(
                        float(update_log["reward"]), accepted_reward_mean, abs_tol=1e-6
                    ),
                    "trainer reward does not match accepted group",
                )
                _require(
                    math.isclose(
                        float(update_log["reward_std"]), accepted_reward_std, abs_tol=1e-6
                    ),
                    "trainer reward std does not match accepted group",
                )
                _require(
                    int(update_log.get("resampling/group_attempts", 0)) == len(attempts),
                    "trainer attempt count does not match rollout audit",
                )
                for signal_type in ("mixed_terminal", "shaping", "zero_variance"):
                    _require(
                        int(
                            update_log.get(f"resampling/accepted_{signal_type}", -1)
                        )
                        == int(accepted_signal_type == signal_type),
                        "trainer accepted-signal metric does not match rollout audit",
                    )
            accepted_records = [
                record
                for record in records
                if record["group_resampling"].get("accepted") is True
            ]
            rejected_records = [
                record
                for record in records
                if record["group_resampling"].get("accepted") is not True
            ]
            def tool_result_counts(
                group_records: list[dict[str, Any]],
            ) -> tuple[int, int]:
                results = [
                    event
                    for record in group_records
                    for event in record["trial"].get("events", [])
                    if event.get("kind") == "tool_result"
                ]
                return len(results), sum(
                    (event.get("payload") or {}).get("ok") is not True
                    for event in results
                )

            proposal_tool_results, proposal_non_ok = tool_result_counts(records)
            accepted_tool_results, accepted_non_ok = tool_result_counts(accepted_records)
            rejected_tool_results, rejected_non_ok = tool_result_counts(rejected_records)
            resampling_summary = {
                "proposal_group_count": sum(len(value) for value in attempts_by_update.values()),
                "proposal_rollout_count": len(records),
                "accepted_rollout_count": len(accepted_records),
                "rejected_rollout_count": len(rejected_records),
                "proposal_success_count": sum(
                    bool(record["trial"]["grade"]["success"]) for record in records
                ),
                "accepted_success_count": sum(
                    bool(record["trial"]["grade"]["success"])
                    for record in accepted_records
                ),
                "rejected_success_count": sum(
                    bool(record["trial"]["grade"]["success"])
                    for record in rejected_records
                ),
                "accepted_signal_counts": dict(sorted(accepted_signal_counts.items())),
                "proposal_tool_result_count": proposal_tool_results,
                "proposal_non_ok_tool_result_count": proposal_non_ok,
                "accepted_tool_result_count": accepted_tool_results,
                "accepted_non_ok_tool_result_count": accepted_non_ok,
                "rejected_tool_result_count": rejected_tool_results,
                "rejected_non_ok_tool_result_count": rejected_non_ok,
                "metric_scope": {
                    "loss_reward_and_reward_std": "accepted_groups",
                    "audit_counters": "all_proposals_split_by_acceptance",
                    "historical_trainer_auxiliary_metrics": (
                        "not_verified_for_acceptance_scope"
                    ),
                },
            }
        expected_ids = {task.id for task in selected_tasks}
        actual_counts = Counter(str(row.get("task_id")) for row in records)
        _require(set(actual_counts) == expected_ids, "rollout tasks do not match selection")
        if config.max_group_attempts == 1 and optimizer_updates == len(selected_tasks):
            _require(
                set(actual_counts.values()) == {config.num_generations},
                "each selected task must have one complete generation group",
            )
        task_by_id = {task.id: task for task in selected_tasks}
        reward_model = RewardModel(version=str(result["reward_model"]["version"]))
        _require(
            asdict(reward_model.weights) == result["reward_model"].get("weights"),
            "reward weights mismatch",
        )
        for row in records:
            task = task_by_id[str(row["task_id"])]
            trial_data = row["trial"]
            grade_data = trial_data["grade"]
            final_state = DeviceState(**trial_data["final_state"])
            events = tuple(TrajectoryEvent(**event) for event in trial_data["events"])
            recomputed_grade = grade_task(
                goal=task.goal,
                final_state=final_state,
                events=events,
            )
            _require(
                recomputed_grade.success == grade_data.get("success")
                and recomputed_grade.checks == grade_data.get("checks"),
                f"rollout grade does not replay: {task.id}",
            )
            trial = TrialResult(
                task_id=trial_data["task_id"],
                agent_name=trial_data["agent_name"],
                termination=trial_data["termination"],
                final_state=final_state,
                events=events,
                grade=recomputed_grade,
                elapsed_ms=trial_data["elapsed_ms"],
            )
            recomputed_reward = json.loads(
                json.dumps(asdict(reward_model.score(task, trial)), sort_keys=True)
            )
            _require(
                recomputed_reward == row.get("reward_trace"),
                f"rollout reward does not replay: {task.id}",
            )
    success_count = sum(bool(row["trial"]["grade"]["success"]) for row in records)
    termination_counts: dict[str, int] = {}
    terminal_message_count = 0
    genuine_length_truncated_count = 0
    loss_masked_count = 0
    termination_mask_record_count = 0
    for row in records:
        termination = str(row["trial"]["termination"])
        termination_counts[termination] = termination_counts.get(termination, 0) + 1
        terminal_message_count += any(
            event.get("kind") == "message"
            and (event.get("payload") or {}).get("kind") in {"answer", "refuse"}
            for event in row["trial"].get("events", [])
        )
        termination_mask = row.get("termination_mask") or {}
        termination_mask_record_count += int(bool(termination_mask))
        genuine_length_truncated_count += int(
            termination_mask.get("length_truncated") is True
        )
        loss_masked_count += int(termination_mask.get("loss_masked") is True)
    _require(rollout.get("success_count") == success_count, "rollout success count mismatch")
    _require(
        rollout.get("termination_counts") == dict(sorted(termination_counts.items())),
        "rollout termination counts mismatch",
    )
    _require(
        rollout.get("terminal_message_count") == terminal_message_count,
        "rollout terminal-message count mismatch",
    )

    guard = result.get("hardware_guard") or {}
    peaks = result.get("peak_vram_mb") or {}
    guard_limit_mb = float(guard.get("max_peak_vram_mb", 0))
    _require(guard_limit_mb > 0, "training result is missing the VRAM guard")
    for metric in ("allocated", "device_used"):
        value = float(peaks.get(metric, math.inf))
        _require(math.isfinite(value), f"invalid peak VRAM metric: {metric}")
        _require(value <= guard_limit_mb, f"training exceeded the VRAM guard: {metric}")

    reasons = []
    if nonzero_reward_std_rate < minimum_nonzero_reward_std_rate:
        reasons.append("reward_variance_rate_below_threshold")
    if nonzero_gradient_rate < minimum_nonzero_gradient_rate:
        reasons.append("gradient_rate_below_threshold")
    source = result.get("source") or {}
    canonical_reasons = []
    if config_path is None:
        canonical_reasons.append("training_config_not_verified")
    if not source.get("commit") or not source.get("tree_sha256"):
        canonical_reasons.append("training_source_not_bound")
    if source.get("dirty") is not False:
        canonical_reasons.append("training_source_not_clean")
    if header.get("source") != source:
        canonical_reasons.append("rollout_source_mismatch")
    if source.get("commit") and source.get("tree_sha256"):
        try:
            historical_tree = _source_tree_sha256_at_commit(root, str(source["commit"]))
        except (OSError, subprocess.SubprocessError, ValueError):
            canonical_reasons.append("training_source_commit_unverifiable")
        else:
            if historical_tree != source["tree_sha256"]:
                canonical_reasons.append("training_source_tree_mismatch")

    evidence = {
        "format_version": "grpo-training-evidence-v0.2",
        "status": "verified",
        "training": {
            "optimizer_updates": optimizer_updates,
            "adapter_sha256": adapter["sha256"],
            "nonzero_reward_std_count": nonzero_reward_std_count,
            "nonzero_reward_std_rate": nonzero_reward_std_rate,
            "nonzero_gradient_count": nonzero_gradient_count,
            "nonzero_gradient_rate": nonzero_gradient_rate,
            # The locked TRL 1.9.2 names this `clipped_ratio`, but it is an EOS/PAD
            # last-token heuristic. Protocol-terminal tool trajectories do not
            # normally emit EOS, so it is not an environment truncation rate.
            "mean_token_no_eos_ratio": mean_token_no_eos_ratio,
            "peak_vram_mb": {
                "allocated": float(peaks["allocated"]),
                "device_used": float(peaks["device_used"]),
                "guard": guard_limit_mb,
            },
        },
        "rollouts": {
            "record_count": len(records),
            "success_count": success_count,
            "success_rate": round(success_count / len(records), 6) if records else 0.0,
            "termination_counts": dict(sorted(termination_counts.items())),
            "terminal_message_count": terminal_message_count,
            "termination_mask_record_count": termination_mask_record_count,
            "genuine_length_truncated_count": (
                genuine_length_truncated_count if termination_mask_record_count else None
            ),
            "loss_masked_count": loss_masked_count if termination_mask_record_count else None,
            "sha256": rollout["sha256"],
        },
        "learning_signal_gate": {
            "status": "passed" if not reasons else "rejected",
            "minimum_nonzero_reward_std_rate": minimum_nonzero_reward_std_rate,
            "minimum_nonzero_gradient_rate": minimum_nonzero_gradient_rate,
            "reasons": reasons,
        },
        "canonical_gate": {
            "status": "passed" if not canonical_reasons else "rejected",
            "reasons": canonical_reasons,
        },
    }
    if resampling_summary is not None:
        evidence["resampling"] = resampling_summary
    return evidence


def _source_tree_sha256_at_commit(project_root: Path, commit: str) -> str:
    prefix = "src/device_agent_arena/"
    listing = subprocess.run(
        ["git", "-C", str(project_root), "ls-tree", "-r", "--name-only", commit, "--", prefix],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.splitlines()
    sources = sorted(path for path in listing if path.endswith(".py"))
    if not sources:
        raise ValueError("commit does not contain the package source tree")
    digest = hashlib.sha256()
    for path in sources:
        payload = subprocess.run(
            ["git", "-C", str(project_root), "show", f"{commit}:{path}"],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        digest.update(path.removeprefix(prefix).encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def verify_rl_evidence(
    *,
    project_root: str | Path,
    task_file_path: str | Path,
    task_selection_path: str | Path | None = None,
    training_config_path: str | Path | None = None,
    training_result_path: str | Path,
    base_run_path: str | Path,
    candidate_run_path: str | Path,
    base_report_path: str | Path,
    candidate_report_path: str | Path,
    comparison_report_path: str | Path,
    minimum_nonzero_reward_std_rate: float = 0.9,
    minimum_nonzero_gradient_rate: float = 0.9,
) -> dict[str, Any]:
    """Recompute an RL pilot's evidence and evaluate a separate promotion gate."""
    if not 0 <= minimum_nonzero_reward_std_rate <= 1:
        raise ValueError("minimum_nonzero_reward_std_rate must be in [0, 1]")
    if not 0 <= minimum_nonzero_gradient_rate <= 1:
        raise ValueError("minimum_nonzero_gradient_rate must be in [0, 1]")

    root = Path(project_root).resolve()
    task_file_path = _resolve_project_file(root, str(task_file_path))
    if task_selection_path is not None:
        task_selection_path = _resolve_project_file(root, str(task_selection_path))
    if training_config_path is not None:
        training_config_path = _resolve_project_file(root, str(training_config_path))
    training_result_path = _resolve_project_file(root, str(training_result_path))
    base_run_path = _resolve_project_file(root, str(base_run_path))
    candidate_run_path = _resolve_project_file(root, str(candidate_run_path))
    base_report_path = _resolve_project_file(root, str(base_report_path))
    candidate_report_path = _resolve_project_file(root, str(candidate_report_path))
    comparison_report_path = _resolve_project_file(root, str(comparison_report_path))
    result = _load_json(training_result_path)
    _require(result.get("status") == "completed", "GRPO training result is not completed")
    optimizer_updates = int(result.get("optimizer_updates", 0))
    _require(optimizer_updates > 0, "GRPO result has no completed optimizer updates")
    config = (
        load_grpo_training_config(training_config_path)
        if training_config_path is not None
        else None
    )
    if config is not None:
        _require(
            grpo_config_sha256(config) == result.get("config_sha256"),
            "config SHA-256 mismatch",
        )
        _require(config.max_updates == optimizer_updates, "config optimizer budget mismatch")
        _require(config.model_id == result.get("model_id"), "config model ID mismatch")
        _require(config.model_revision == result.get("model_revision"), "config revision mismatch")
        _require(config.output_dir == result.get("output_dir"), "config output path mismatch")

    adapter = result.get("adapter") or {}
    adapter_path = _resolve_project_file(root, str(adapter.get("path", "")))
    _require(adapter_path.stat().st_size == adapter.get("size_bytes"), "adapter size mismatch")
    adapter_sha256 = _sha256(adapter_path)
    _require(adapter_sha256 == adapter.get("sha256"), "adapter SHA-256 mismatch")

    initial_adapter_sha256 = None
    if result.get("initial_adapter") is not None:
        initial_adapter_root = (root / str(result["initial_adapter"])).resolve()
        _require(
            initial_adapter_root.is_relative_to(root),
            "initial adapter path escapes project root",
        )
        initial_adapter_model = _resolve_project_file(
            root,
            str(Path(result["initial_adapter"]) / "adapter_model.safetensors"),
        )
        initial_adapter_config = _resolve_project_file(
            root,
            str(Path(result["initial_adapter"]) / "adapter_config.json"),
        )
        initial_provenance = result.get("initial_adapter_provenance") or {}
        initial_adapter_sha256 = _sha256(initial_adapter_model)
        _require(
            initial_adapter_sha256 == initial_provenance.get("model_sha256"),
            "initial adapter SHA-256 mismatch",
        )
        _require(
            initial_adapter_model.stat().st_size == initial_provenance.get("model_size_bytes"),
            "initial adapter size mismatch",
        )
        _require(
            _sha256(initial_adapter_config) == initial_provenance.get("config_sha256"),
            "initial adapter config SHA-256 mismatch",
        )

    dataset = result.get("dataset") or {}
    dataset_path = _resolve_project_file(root, str(dataset.get("path", "")))
    _require(_sha256(dataset_path) == dataset.get("sha256"), "training data SHA-256 mismatch")
    if config is not None:
        _require(config.train_data == dataset.get("path"), "config training-data path mismatch")
        _require(
            config.initial_adapter == result.get("initial_adapter"),
            "config initial-adapter path mismatch",
        )

    trainer_state_path = _resolve_project_file(root, str(result.get("trainer_state", "")))
    trainer_state = _load_json(trainer_state_path)
    _require(
        trainer_state.get("global_step") == optimizer_updates,
        "trainer_state global_step does not match the training result",
    )
    _require(
        trainer_state.get("max_steps") == optimizer_updates,
        "training stopped before or after the declared optimizer budget",
    )
    update_logs = _finite_update_logs(trainer_state, optimizer_updates)
    nonzero_reward_std_count = sum(float(row["reward_std"]) > 0 for row in update_logs)
    nonzero_reward_std_rate = round(nonzero_reward_std_count / optimizer_updates, 6)
    nonzero_gradient_count = sum(float(row["grad_norm"]) > 0 for row in update_logs)
    nonzero_gradient_rate = round(nonzero_gradient_count / optimizer_updates, 6)
    clipped_ratios = [
        float(row["completions/clipped_ratio"])
        for row in update_logs
        if row.get("completions/clipped_ratio") is not None
    ]
    mean_completion_clipped_ratio = (
        round(sum(clipped_ratios) / len(clipped_ratios), 6) if clipped_ratios else None
    )
    learning_signal_reasons = []
    if nonzero_reward_std_rate < minimum_nonzero_reward_std_rate:
        learning_signal_reasons.append("reward_variance_rate_below_threshold")
    if nonzero_gradient_rate < minimum_nonzero_gradient_rate:
        learning_signal_reasons.append("gradient_rate_below_threshold")

    guard = result.get("hardware_guard") or {}
    peaks = result.get("peak_vram_mb") or {}
    guard_limit_mb = float(guard.get("max_peak_vram_mb", 0))
    _require(guard_limit_mb > 0, "training result is missing the VRAM guard")
    for metric in ("allocated", "device_used"):
        value = float(peaks.get(metric, math.inf))
        _require(math.isfinite(value), f"training result has invalid peak VRAM metric: {metric}")
        _require(value <= guard_limit_mb, f"training exceeded the VRAM guard: {metric}")

    reward_model = result.get("reward_model") or {}
    _require(reward_model.get("version"), "training result is missing reward-model version")
    _require(
        isinstance(reward_model.get("weights"), dict) and reward_model["weights"],
        "training result is missing reward weights",
    )
    dependency_lock_sha256 = result.get("dependency_lock_sha256")
    if dependency_lock_sha256 is not None:
        lock_path = _resolve_project_file(root, "uv.lock")
        _require(_sha256(lock_path) == dependency_lock_sha256, "dependency lock SHA-256 mismatch")

    stored_base_report = _load_json(base_report_path)
    stored_candidate_report = _load_json(candidate_report_path)
    base_run_validation = verify_run_grades(
        task_file_path=task_file_path,
        run_path=base_run_path,
        task_selection_path=task_selection_path,
    )
    candidate_run_validation = verify_run_grades(
        task_file_path=task_file_path,
        run_path=candidate_run_path,
        task_selection_path=task_selection_path,
    )
    recomputed_base_report = build_report(base_run_path)
    recomputed_candidate_report = build_report(candidate_run_path)
    _require(stored_base_report == recomputed_base_report, "base report does not match raw run")
    _require(
        stored_candidate_report == recomputed_candidate_report,
        "candidate report does not match raw run",
    )
    recomputed_comparison = compare_reports(recomputed_base_report, recomputed_candidate_report)
    _require(
        _load_json(comparison_report_path) == recomputed_comparison,
        "comparison report does not match recomputed reports",
    )
    candidate_backend = (recomputed_candidate_report.get("agent_config") or {}).get(
        "backend", {}
    )
    _require(
        candidate_backend.get("adapter_sha256") == adapter_sha256,
        "candidate run does not use the verified GRPO adapter",
    )
    if initial_adapter_sha256 is not None:
        base_backend = (recomputed_base_report.get("agent_config") or {}).get("backend", {})
        _require(
            base_backend.get("adapter_sha256") == initial_adapter_sha256,
            "base run does not use the GRPO warm-start adapter",
        )

    task_success_improved = recomputed_comparison["task_success_delta_pp"] > 0
    missing_parameter = recomputed_comparison["by_category"].get("missing_parameter")
    clarification_not_worse = bool(
        missing_parameter is None or missing_parameter["candidate"] >= missing_parameter["base"]
    )
    promotion_reasons = []
    if learning_signal_reasons:
        promotion_reasons.append("learning_signal_gate_rejected")
    if not task_success_improved:
        promotion_reasons.append("task_success_not_improved")
    if not clarification_not_worse:
        promotion_reasons.append("missing_parameter_regressed")

    source = result.get("source") or {}
    limitations = []
    if not source.get("commit") or not source.get("tree_sha256"):
        limitations.append("training_result_missing_source_provenance")
    run_sources = (base_run_validation["source"], candidate_run_validation["source"])
    if any(run_source.get("dirty") is not False for run_source in run_sources):
        limitations.append("evaluation_run_not_clean")
    canonical_reasons = []
    if "training_result_missing_source_provenance" in limitations:
        canonical_reasons.append("training_source_not_bound")
    if "evaluation_run_not_clean" in limitations:
        canonical_reasons.append("evaluation_source_not_clean")

    return {
        "format_version": "rl-evidence-v0.2",
        "status": "verified",
        "training": {
            "optimizer_updates": optimizer_updates,
            "adapter_sha256": adapter_sha256,
            "initial_adapter_sha256": initial_adapter_sha256,
            "dataset_sha256": dataset["sha256"],
            "reward_model_version": reward_model["version"],
            "config_sha256": result.get("config_sha256"),
            "nonzero_reward_std_count": nonzero_reward_std_count,
            "nonzero_reward_std_rate": nonzero_reward_std_rate,
            "nonzero_gradient_count": nonzero_gradient_count,
            "nonzero_gradient_rate": nonzero_gradient_rate,
            "mean_completion_clipped_ratio": mean_completion_clipped_ratio,
            "peak_vram_mb": {
                "allocated": float(peaks["allocated"]),
                "device_used": float(peaks["device_used"]),
                "guard": guard_limit_mb,
            },
        },
        "evaluation": {
            "task_count": recomputed_comparison["task_count"],
            "base_task_success_rate": recomputed_comparison["base_task_success_rate"],
            "candidate_task_success_rate": recomputed_comparison[
                "candidate_task_success_rate"
            ],
            "task_success_delta_pp": recomputed_comparison["task_success_delta_pp"],
            "task_selection_sha256": base_run_validation["task_selection_sha256"],
            "missing_parameter": missing_parameter,
            "source_commit": recomputed_comparison["source_commit"],
            "source_tree_sha256": recomputed_comparison["source_tree_sha256"],
            "base_source_dirty": base_run_validation["source"].get("dirty"),
            "candidate_source_dirty": candidate_run_validation["source"].get("dirty"),
        },
        "learning_signal_gate": {
            "status": "passed" if not learning_signal_reasons else "rejected",
            "minimum_nonzero_reward_std_rate": minimum_nonzero_reward_std_rate,
            "minimum_nonzero_gradient_rate": minimum_nonzero_gradient_rate,
            "reasons": learning_signal_reasons,
        },
        "promotion_gate": {
            "status": "passed" if not promotion_reasons else "rejected",
            "reasons": promotion_reasons,
        },
        "canonical_gate": {
            "status": "passed" if not canonical_reasons else "rejected",
            "reasons": canonical_reasons,
        },
        "limitations": limitations,
    }


def write_rl_evidence(*, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    evidence = verify_rl_evidence(**kwargs)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        (json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    return evidence


def write_grpo_training_evidence(
    *, output_path: str | Path, **kwargs: Any
) -> dict[str, Any]:
    evidence = verify_grpo_training_evidence(**kwargs)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        (json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    return evidence
