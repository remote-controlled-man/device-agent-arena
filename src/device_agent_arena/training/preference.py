from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from device_agent_arena.agents.hf_backend import normalize_functiongemma_messages
from device_agent_arena.artifacts.run_io import load_run, task_suite_payload
from device_agent_arena.dataset.rl_suite import load_rl_task_selection
from device_agent_arena.domain.models import (
    MessageAction,
    TaskSpec,
    ToolCall,
    ToolCallBatch,
    TrialResult,
)
from device_agent_arena.domain.task_io import load_tasks
from device_agent_arena.harness.environment import AgentEnvironment
from device_agent_arena.harness.types import HarnessConfig
from device_agent_arena.simulator.tools import tool_schemas
from device_agent_arena.training.oec import (
    _action_completion,
    _context_messages,
    _CorrectiveOracle,
    _heldout_entities,
    _multi_seam_repairs,
    _sha256,
    _student_actions,
)

PREFERENCE_FORMAT_VERSION = "fctr-preference-v0.1"
PREFERENCE_MANIFEST_VERSION = "fctr-preference-manifest-v0.1"
PREFERENCE_AUDIT_VERSION = "fctr-preference-audit-v0.1"

AgentAction = ToolCall | ToolCallBatch | MessageAction


def _jsonl_payload(rows: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    ).encode("utf-8")


def _counterfactual_trial(
    task: TaskSpec,
    student_actions: list[AgentAction],
    prefix_count: int,
    first_action: AgentAction,
) -> TrialResult:
    environment = AgentEnvironment(HarnessConfig(max_steps=10))
    context = environment.reset(task)
    for action in student_actions[:prefix_count]:
        transition = environment.step(action)
        context = transition.observation
        if transition.done:
            return environment.finish("fctr-preference-invalid-prefix")
    transition = environment.step(first_action)
    context = transition.observation
    if not transition.done:
        oracle = _CorrectiveOracle(task, context)
        while True:
            transition = environment.step(oracle.act(context))
            context = transition.observation
            if transition.done:
                break
    return environment.finish("fctr-preference-counterfactual")


def _counterfactual_metric(trial: TrialResult, prefix_count: int) -> tuple[int, int, int]:
    non_ok = sum(
        event.kind == "tool_result"
        and event.payload.get("ok") is not True
        and event.step > prefix_count
        for event in trial.events
    )
    final_step = max((event.step for event in trial.events), default=prefix_count)
    return (int(trial.grade.success), -non_ok, -(final_step - prefix_count))


def _load_inputs(
    task_path: Path,
    task_selection_path: Path | None,
    student_run_path: Path,
) -> tuple[tuple[TaskSpec, ...], list[dict[str, Any]]]:
    pool = load_tasks(task_path)
    tasks = pool
    if task_selection_path is not None:
        tasks = load_rl_task_selection(
            selection_path=task_selection_path,
            source_task_path=task_path,
            tasks=pool,
            split="train",
        )
    header, trials = load_run(student_run_path)
    if header.get("split") != "train" or any(task.split != "train" for task in tasks):
        raise ValueError("FCTR preferences accept train-split inputs only")
    expected_suite = hashlib.sha256(task_suite_payload(tasks)).hexdigest()
    if header.get("task_suite_sha256") != expected_suite:
        raise ValueError("student run task suite does not match preference tasks")
    if [str(row["task_id"]) for row in trials] != [task.id for task in tasks]:
        raise ValueError("student run task order does not match preference tasks")
    return tasks, trials


def _build_preferences(
    *,
    tasks: tuple[TaskSpec, ...],
    trials: list[dict[str, Any]],
    protocol: str = "openai",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if protocol not in {"openai", "functiongemma"}:
        raise ValueError("unsupported FCTR preference protocol")
    task_by_id = {task.id: task for task in tasks}
    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    counts = {"candidate_seam_count": 0, "different_action_count": 0}
    for trial_data in trials:
        if trial_data.get("grade", {}).get("success") is True:
            continue
        task = task_by_id[str(trial_data["task_id"])]
        student_actions = _student_actions(trial_data)
        for seam_type, _trial, prefix_count, expert_decisions in _multi_seam_repairs(
            task, trial_data
        ):
            counts["candidate_seam_count"] += 1
            pair_id = f"{task.id}:{seam_type}:prefix-{prefix_count}"
            base_audit = {
                "record_type": "preference_pair",
                "pair_id": pair_id,
                "task_id": task.id,
                "seam_type": seam_type,
                "student_prefix_action_count": prefix_count,
            }
            if prefix_count >= len(student_actions):
                audit.append({**base_audit, "accepted": False, "reason": "no_student_next_action"})
                continue
            context, chosen_action = expert_decisions[0]
            rejected_action = student_actions[prefix_count]
            chosen = [_action_completion(chosen_action, call_id="fctr_preference")]
            rejected = [_action_completion(rejected_action, call_id="fctr_preference")]
            if chosen == rejected:
                audit.append({**base_audit, "accepted": False, "reason": "identical_action"})
                continue
            counts["different_action_count"] += 1
            chosen_trial = _counterfactual_trial(
                task, student_actions, prefix_count, chosen_action
            )
            rejected_trial = _counterfactual_trial(
                task, student_actions, prefix_count, rejected_action
            )
            chosen_metric = _counterfactual_metric(chosen_trial, prefix_count)
            rejected_metric = _counterfactual_metric(rejected_trial, prefix_count)
            if chosen_metric <= rejected_metric:
                audit.append(
                    {
                        **base_audit,
                        "accepted": False,
                        "reason": "counterfactual_not_strictly_better",
                        "chosen_counterfactual_metric": list(chosen_metric),
                        "rejected_counterfactual_metric": list(rejected_metric),
                    }
                )
                continue
            prompt = _context_messages(context)
            if protocol == "functiongemma":
                chosen_conversation = normalize_functiongemma_messages([*prompt, *chosen])
                rejected_conversation = normalize_functiongemma_messages(
                    [*prompt, *rejected]
                )
                prompt_length = len(prompt)
                chosen_prompt = chosen_conversation[:prompt_length]
                rejected_prompt = rejected_conversation[:prompt_length]
                if chosen_prompt != rejected_prompt:
                    raise ValueError("preference branches must share an identical prompt")
                prompt = chosen_prompt
                chosen = chosen_conversation[prompt_length:]
                rejected = rejected_conversation[prompt_length:]
            canonical = json.dumps(
                {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                ensure_ascii=False,
                sort_keys=True,
            )
            canonical_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if canonical_hash in seen_hashes:
                audit.append(
                    {
                        **base_audit,
                        "accepted": False,
                        "reason": "duplicate_preference_pair",
                        "canonical_hash": canonical_hash,
                    }
                )
                continue
            seen_hashes.add(canonical_hash)
            metadata = {
                "generator": PREFERENCE_FORMAT_VERSION,
                "source_split": "train",
                "pair_id": pair_id,
                "task_id": task.id,
                "seam_type": seam_type,
                "student_prefix_action_count": prefix_count,
                "chosen_counterfactual_metric": list(chosen_metric),
                "rejected_counterfactual_metric": list(rejected_metric),
                "canonical_hash": canonical_hash,
                "protocol": protocol,
            }
            records.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "tools": tool_schemas(include_interaction=True),
                    "metadata": metadata,
                }
            )
            audit.append({**base_audit, "accepted": True, **metadata})
    return records, audit, counts


def export_fctr_preferences(
    *,
    task_path: str | Path,
    task_selection_path: str | Path | None,
    student_run_path: str | Path,
    heldout_data_root: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
    protocol: str = "openai",
) -> dict[str, Any]:
    task_source = Path(task_path)
    selection_source = (
        Path(task_selection_path) if task_selection_path is not None else None
    )
    run_source = Path(student_run_path)
    destinations = (Path(output_path), Path(manifest_path), Path(audit_path))
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite FCTR preference artifact: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
    tasks, trials = _load_inputs(task_source, selection_source, run_source)
    records, audit_rows, counts = _build_preferences(
        tasks=tasks, trials=trials, protocol=protocol
    )
    dataset_payload = _jsonl_payload(records)
    audit_header = {
        "record_type": "preference_audit_header",
        "format_version": PREFERENCE_AUDIT_VERSION,
        "task_sha256": _sha256(task_source),
        "task_selection_sha256": (
            _sha256(selection_source) if selection_source is not None else None
        ),
        "student_run_sha256": _sha256(run_source),
        "record_count": len(audit_rows),
    }
    audit_payload = _jsonl_payload([audit_header, *audit_rows])
    destinations[0].write_bytes(dataset_payload)
    destinations[2].write_bytes(audit_payload)
    heldout_instructions, heldout_entities = _heldout_entities(Path(heldout_data_root))
    prompt_instructions = {str(row["prompt"][1]["content"]) for row in records}
    completion_text = json.dumps(
        [[row["chosen"], row["rejected"]] for row in records], ensure_ascii=False
    )
    manifest = {
        "format_version": PREFERENCE_MANIFEST_VERSION,
        "generator": PREFERENCE_FORMAT_VERSION,
        "protocol": protocol,
        "task_sha256": _sha256(task_source),
        "task_selection_sha256": (
            _sha256(selection_source) if selection_source is not None else None
        ),
        "student_run_sha256": _sha256(run_source),
        "dataset_sha256": hashlib.sha256(dataset_payload).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_payload).hexdigest(),
        "pair_count": len(records),
        **counts,
        "unique_pair_count": len({row["metadata"]["canonical_hash"] for row in records}),
        "seam_type_counts": dict(
            sorted(Counter(row["metadata"]["seam_type"] for row in records).items())
        ),
        "heldout_instruction_overlap_count": len(prompt_instructions & heldout_instructions),
        "heldout_completion_entities": sorted(
            entity for entity in heldout_entities if entity and entity in completion_text
        ),
    }
    destinations[1].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_fctr_preferences(
    *,
    task_path: str | Path,
    task_selection_path: str | Path | None,
    student_run_path: str | Path,
    heldout_data_root: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
    protocol: str = "openai",
) -> dict[str, Any]:
    task_source = Path(task_path)
    selection_source = (
        Path(task_selection_path) if task_selection_path is not None else None
    )
    run_source = Path(student_run_path)
    dataset_source = Path(dataset_path)
    audit_source = Path(audit_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("format_version") != PREFERENCE_MANIFEST_VERSION:
        raise ValueError("unsupported FCTR preference manifest format")
    if manifest.get("protocol", "openai") != protocol:
        raise ValueError("FCTR preference protocol does not match manifest")
    sources = [
        ("task", task_source),
        ("student_run", run_source),
        ("dataset", dataset_source),
        ("audit", audit_source),
    ]
    if selection_source is not None:
        sources.append(("task_selection", selection_source))
    elif manifest.get("task_selection_sha256") is not None:
        raise ValueError("task selection is required by the preference manifest")
    for label, source in sources:
        if manifest.get(f"{label}_sha256") != _sha256(source):
            raise ValueError(f"{label.replace('_', ' ')} SHA-256 mismatch")
    tasks, trials = _load_inputs(task_source, selection_source, run_source)
    expected_records, expected_audit, counts = _build_preferences(
        tasks=tasks, trials=trials, protocol=protocol
    )
    actual_records = [
        json.loads(line) for line in dataset_source.read_text(encoding="utf-8").splitlines()
    ]
    audit_rows = [
        json.loads(line) for line in audit_source.read_text(encoding="utf-8").splitlines()
    ]
    if not audit_rows or audit_rows[0].get("format_version") != PREFERENCE_AUDIT_VERSION:
        raise ValueError("unsupported FCTR preference audit format")
    if audit_rows[1:] != expected_audit:
        raise ValueError("FCTR preference audit does not replay")
    if actual_records != expected_records:
        raise ValueError("FCTR preference records do not replay")
    if manifest.get("pair_count") != len(expected_records) or any(
        manifest.get(key) != value for key, value in counts.items()
    ):
        raise ValueError("FCTR preference manifest counts do not replay")
    heldout_instructions, heldout_entities = _heldout_entities(Path(heldout_data_root))
    instructions = {str(row["prompt"][1]["content"]) for row in actual_records}
    text = json.dumps(
        [[row["chosen"], row["rejected"]] for row in actual_records],
        ensure_ascii=False,
    )
    if instructions & heldout_instructions or any(
        entity and entity in text for entity in heldout_entities
    ):
        raise ValueError("FCTR preference data leaks held-out content")
    return {
        "status": "verified",
        "protocol": protocol,
        "replayed_pair_count": len(expected_records),
        "candidate_seam_count": counts["candidate_seam_count"],
        "different_action_count": counts["different_action_count"],
    }
