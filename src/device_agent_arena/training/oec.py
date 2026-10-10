from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from device_agent_arena.agents.oracle import OracleAgent
from device_agent_arena.artifacts.run_io import load_run, task_suite_payload
from device_agent_arena.dataset.rl_suite import (
    load_rl_task_selection,
    write_rl_task_selection,
)
from device_agent_arena.domain.models import (
    MessageAction,
    TaskSpec,
    ToolCall,
    ToolCallBatch,
    TrajectoryEvent,
    TrialResult,
)
from device_agent_arena.domain.task_io import load_tasks
from device_agent_arena.grading.task_grader import grade_task
from device_agent_arena.harness.environment import AgentEnvironment
from device_agent_arena.harness.types import AgentContext, HarnessConfig
from device_agent_arena.prompts import BASE_SYSTEM_PROMPT
from device_agent_arena.simulator.tools import tool_schemas

OEC_FORMAT_VERSION = "oec-fctr-v0.1"
OEC_MULTI_FORMAT_VERSION = "oec-fctr-v0.2"
OEC_AUDIT_FORMAT_VERSION = "oec-repair-audit-v0.1"
OEC_MANIFEST_FORMAT_VERSION = "oec-sft-manifest-v0.1"
OEC_ABLATION_FORMAT_VERSION = "oec-ablation-manifest-v0.1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_oec_train_tasks(
    tasks: tuple[TaskSpec, ...] | list[TaskSpec], *, count: int = 32
) -> tuple[TaskSpec, ...]:
    """Deterministically interleave v0.3 structures for an OEC collection subset."""
    task_list = tuple(tasks)
    if count < 1 or count > len(task_list):
        raise ValueError("OEC selection count must be within the train task pool")
    if any(task.split != "train" for task in task_list):
        raise ValueError("OEC selection accepts train tasks only")
    buckets: dict[str, deque[TaskSpec]] = defaultdict(deque)
    for task in task_list:
        structure = str(task.provenance.get("structure", ""))
        if not structure:
            raise ValueError("OEC task is missing a structure label")
        buckets[structure].append(task)
    selected: list[TaskSpec] = []
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].popleft())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError("OEC selection could not satisfy its requested count")
    return tuple(selected)


def write_oec_task_selection(
    *,
    source_task_path: str | Path,
    output_path: str | Path,
    tasks: tuple[TaskSpec, ...] | list[TaskSpec] | None = None,
    count: int = 32,
) -> dict[str, Any]:
    source = Path(source_task_path)
    task_pool = tuple(tasks) if tasks is not None else load_tasks(source)
    if tasks is not None and not source.exists():
        from device_agent_arena.domain.task_io import write_tasks

        write_tasks(source, list(task_pool))
    selected = select_oec_train_tasks(task_pool, count=count)
    return write_rl_task_selection(
        source_task_path=source,
        output_path=output_path,
        split="train",
        task_ids=[task.id for task in selected],
        selection_policy="oec-train-structure-round-robin-v0.1",
    )


def _jsonl_payload(rows: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    ).encode("utf-8")


def _load_bound_records(path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _sha256(path)
    if digest not in {manifest.get("sha256"), manifest.get("dataset_sha256")}:
        raise ValueError(f"dataset SHA-256 does not match manifest: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if manifest.get("record_count") != len(records):
        raise ValueError(f"dataset record count does not match manifest: {path}")
    hashes = [str(record.get("metadata", {}).get("canonical_hash", "")) for record in records]
    if not all(hashes):
        raise ValueError(f"dataset canonical hashes must be present: {path}")
    unique_by_hash = {
        record["metadata"]["canonical_hash"]: record for record in records
    }
    unique_records = list(unique_by_hash.values())
    declared_unique = manifest.get("unique_supervision_count", len(unique_records))
    if declared_unique != len(unique_records):
        raise ValueError(f"dataset unique supervision count does not match manifest: {path}")
    return unique_records


def _decision_name(record: dict[str, Any]) -> str:
    completion = record.get("completion", [{}])[0]
    calls = completion.get("tool_calls", [])
    return str(calls[0].get("function", {}).get("name", "message")) if calls else "message"


def _stratified_sample(
    records: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if count < 1 or count > len(records):
        raise ValueError("ablation sample count must fit the unique source pool")
    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    rng = random.Random(seed)
    for key in sorted({_decision_name(record) for record in records}):
        bucket = [record for record in records if _decision_name(record) == key]
        rng.shuffle(bucket)
        buckets[key].extend(bucket)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].popleft())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError("could not fill compute-matched ablation sample")
    return selected


def _seam_first_sample(
    records: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    first = [
        record
        for record in records
        if record.get("metadata", {}).get("expert_turn") == 1
        and record.get("metadata", {}).get("repair_id")
    ]
    first_by_repair = {
        str(record["metadata"]["repair_id"]): record for record in first
    }
    if len(first_by_repair) > count:
        raise ValueError("ablation count cannot retain every first corrective seam")
    selected = list(first_by_repair.values())
    selected_hashes = {
        str(record["metadata"]["canonical_hash"]) for record in selected
    }
    remainder = [
        record
        for record in records
        if str(record["metadata"]["canonical_hash"]) not in selected_hashes
    ]
    selected.extend(
        _stratified_sample(remainder, count=count - len(selected), seed=seed)
    )
    return selected


def build_oec_ablation_datasets(
    *,
    correction_path: str | Path,
    correction_manifest_path: str | Path,
    replay_path: str | Path,
    replay_manifest_path: str | Path,
    output_root: str | Path,
    record_count: int = 50,
    seed: int = 42,
    prioritize_first_seams: bool = False,
) -> dict[str, Any]:
    """Build unique, equal-size static/FCTR/mixed SFT ablations."""
    correction_source = Path(correction_path)
    correction_manifest_source = Path(correction_manifest_path)
    replay_source = Path(replay_path)
    replay_manifest_source = Path(replay_manifest_path)
    destination = Path(output_root)
    corrections = _load_bound_records(correction_source, correction_manifest_source)
    replay = _load_bound_records(replay_source, replay_manifest_source)
    if record_count > len(corrections) or record_count > len(replay):
        raise ValueError("each ablation source must contain enough unique records")
    fctr_count = record_count // 2
    replay_count = record_count - fctr_count
    sample_corrections = _seam_first_sample if prioritize_first_seams else _stratified_sample
    routes = {
        "static_replay": _stratified_sample(replay, count=record_count, seed=seed),
        "fctr_only": sample_corrections(corrections, count=record_count, seed=seed),
        "fctr_replay": [
            *sample_corrections(corrections, count=fctr_count, seed=seed),
            *_stratified_sample(replay, count=replay_count, seed=seed),
        ],
    }
    source_hashes = {
        "fctr": _sha256(correction_source),
        "fctr_manifest": _sha256(correction_manifest_source),
        "static_replay": _sha256(replay_source),
        "static_replay_manifest": _sha256(replay_manifest_source),
    }
    receipt: dict[str, Any] = {
        "format_version": OEC_ABLATION_FORMAT_VERSION,
        "seed": seed,
        "record_count_per_route": record_count,
        "prioritize_first_seams": prioritize_first_seams,
        "sources": source_hashes,
        "routes": {},
    }
    destination.mkdir(parents=True, exist_ok=True)
    for route, records in routes.items():
        hashes = [record["metadata"]["canonical_hash"] for record in records]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"{route} contains duplicate supervision")
        data_path = destination / f"{route}.jsonl"
        manifest_path = destination / f"manifest_{route}.json"
        if data_path.exists() or manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite OEC ablation route: {route}")
        payload = _jsonl_payload(records)
        data_path.write_bytes(payload)
        source_counts = {
            "fctr": sum(
                record.get("metadata", {}).get("generator")
                in {OEC_FORMAT_VERSION, OEC_MULTI_FORMAT_VERSION}
                for record in records
            ),
        }
        source_counts["static_replay"] = len(records) - source_counts["fctr"]
        route_manifest = {
            "format_version": OEC_ABLATION_FORMAT_VERSION,
            "generator": f"oec-ablation-{route}-v0.1",
            "route": route,
            "seed": seed,
            "record_count": len(records),
            "unique_supervision_count": len(set(hashes)),
            "source_counts": source_counts,
            "source_sha256": source_hashes,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        manifest_path.write_text(
            json.dumps(route_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt["routes"][route] = {
            "data_path": data_path.name,
            "manifest_path": manifest_path.name,
            "record_count": len(records),
            "source_counts": source_counts,
            "sha256": route_manifest["sha256"],
        }
    (destination / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def _replayable_trial_payload(trial: TrialResult) -> dict[str, Any]:
    payload = asdict(trial)
    # Wall-clock time is telemetry, not deterministic trajectory semantics.
    payload.pop("elapsed_ms", None)
    return json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _assistant_call(call: ToolCall, call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments, ensure_ascii=False, sort_keys=True
                    ),
                },
            }
        ],
    }


def _action_completion(
    action: ToolCall | ToolCallBatch | MessageAction,
    *,
    call_id: str,
) -> dict[str, Any]:
    if isinstance(action, ToolCall):
        return _assistant_call(action, call_id)
    if isinstance(action, ToolCallBatch):
        calls = [
            _assistant_call(call, f"{call_id}_{index}")["tool_calls"][0]
            for index, call in enumerate(action.calls, 1)
        ]
        return {"role": "assistant", "content": "", "tool_calls": calls}
    return {"role": "assistant", "content": action.content}


def _tool_message(event: TrajectoryEvent) -> dict[str, Any]:
    payload = event.payload
    normalized = {
        "name": payload["name"],
        "ok": bool(payload["ok"]),
        "code": payload["code"],
        "message": payload["message"],
        "state_delta": payload.get("state_delta", {}),
        "data": payload.get("data", {}),
    }
    return {
        "role": "tool",
        "name": payload["name"],
        "tool_call_id": payload["id"],
        "content": json.dumps(normalized, ensure_ascii=False, sort_keys=True),
    }


def _context_messages(context: AgentContext) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": context.instruction},
    ]
    for event in context.events:
        if event.kind == "tool_call":
            messages.append(
                _assistant_call(
                    ToolCall(
                        str(event.payload["name"]),
                        dict(event.payload.get("arguments", {})),
                    ),
                    str(event.payload["id"]),
                )
            )
        elif event.kind == "tool_result":
            messages.append(_tool_message(event))
        elif event.kind == "user":
            messages.append(
                {"role": "user", "content": str(event.payload.get("content", ""))}
            )
        elif event.kind == "message" and not event.payload.get("via_tool"):
            messages.append(
                {"role": "assistant", "content": str(event.payload.get("content", ""))}
            )
    return messages


def _student_actions(trial: dict[str, Any]) -> list[ToolCall | ToolCallBatch | MessageAction]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for event in trial.get("events", []):
        if event.get("kind") in {"tool_call", "message"}:
            if event.get("kind") == "message" and event.get("payload", {}).get("via_tool"):
                continue
            grouped.setdefault(int(event["step"]), []).append(event)
    actions: list[ToolCall | ToolCallBatch | MessageAction] = []
    for events in grouped.values():
        tool_events = [event for event in events if event["kind"] == "tool_call"]
        if tool_events:
            calls = tuple(
                ToolCall(
                    str(event["payload"]["name"]),
                    dict(event["payload"].get("arguments", {})),
                )
                for event in tool_events
            )
            actions.append(calls[0] if len(calls) == 1 else ToolCallBatch(calls))
            continue
        event = events[0]
        payload = event["payload"]
        actions.append(
            MessageAction(
                kind=str(payload["kind"]),  # type: ignore[arg-type]
                content=str(payload.get("content", "")),
                field=payload.get("field"),
            )
        )
    return actions


def _is_terminal_action(action: ToolCall | ToolCallBatch | MessageAction) -> bool:
    if isinstance(action, ToolCall):
        return action.name == "finish_task"
    if isinstance(action, ToolCallBatch):
        return any(call.name == "finish_task" for call in action.calls)
    return action.kind in {"answer", "refuse"}


class _CorrectiveOracle:
    name = "corrective-oracle-v0.1"

    def __init__(self, task: TaskSpec, context: AgentContext) -> None:
        successful_tools = {
            str(event.payload.get("name"))
            for event in context.events
            if event.kind == "tool_result" and event.payload.get("ok") is True
        }
        clarified_fields = {
            str(event.payload.get("field"))
            for event in context.events
            if event.kind == "message" and event.payload.get("kind") == "clarify"
        }
        actions = []
        for action in OracleAgent._build_actions(task, interaction_tools=True):
            if not isinstance(action, ToolCall):
                actions.append(action)
            elif action.name == "request_clarification":
                if str(action.arguments.get("field")) not in clarified_fields:
                    actions.append(action)
            elif action.name not in successful_tools:
                actions.append(action)
        self._task = task
        self._actions = deque(actions)
        self._last_tool_call: ToolCall | None = None

    def act(self, context: AgentContext) -> ToolCall | MessageAction:
        last_result = next(
            (event for event in reversed(context.events) if event.kind == "tool_result"),
            None,
        )
        if (
            last_result is not None
            and last_result.payload.get("ok") is not True
            and last_result.payload.get("code") == "transient_failure"
            and self._last_tool_call is not None
        ):
            return self._last_tool_call
        if self._actions:
            action = self._actions.popleft()
            if isinstance(action, ToolCall):
                self._last_tool_call = action
            return action
        return ToolCall(
            "finish_task",
            {
                "kind": self._task.goal.terminal_behavior,
                "content": "设备配置已完成。",
            },
        )


def _repair_trial(
    task: TaskSpec,
    student_actions: list[ToolCall | ToolCallBatch | MessageAction],
) -> (
    tuple[
        TrialResult,
        int,
        list[tuple[AgentContext, ToolCall | ToolCallBatch | MessageAction]],
    ]
    | None
):
    nonterminal_count = len(student_actions)
    while nonterminal_count and _is_terminal_action(student_actions[nonterminal_count - 1]):
        nonterminal_count -= 1
    for prefix_count in range(nonterminal_count, -1, -1):
        repaired = _repair_at_prefix(task, student_actions, prefix_count)
        if repaired is not None:
            return repaired
    return None


def _repair_at_prefix(
    task: TaskSpec,
    student_actions: list[ToolCall | ToolCallBatch | MessageAction],
    prefix_count: int,
) -> (
    tuple[
        TrialResult,
        int,
        list[tuple[AgentContext, ToolCall | ToolCallBatch | MessageAction]],
    ]
    | None
):
    if prefix_count < 0 or prefix_count > len(student_actions):
        raise ValueError("repair prefix count is outside the student trajectory")
    environment = AgentEnvironment(HarnessConfig(max_steps=10))
    context = environment.reset(task)
    for action in student_actions[:prefix_count]:
        transition = environment.step(action)
        context = transition.observation
        if transition.done:
            return None
    oracle = _CorrectiveOracle(task, context)
    expert_decisions = []
    while True:
        action = oracle.act(context)
        expert_decisions.append((context, action))
        transition = environment.step(action)
        if transition.done:
            trial = environment.finish(oracle.name)
            break
        context = transition.observation
    return (trial, prefix_count, expert_decisions) if trial.grade.success else None


def _multi_seam_repairs(
    task: TaskSpec,
    trial_data: dict[str, Any],
) -> list[
    tuple[
        str,
        TrialResult,
        int,
        list[tuple[AgentContext, ToolCall | ToolCallBatch | MessageAction]],
    ]
]:
    actions = _student_actions(trial_data)
    candidates: list[tuple[str, int | None]] = []
    if task.goal.required_clarifications:
        candidates.append(("clarification_grounding", 0))
    failure_steps = [
        int(event["step"])
        for event in trial_data.get("events", [])
        if event.get("kind") == "tool_result"
        and event.get("payload", {}).get("ok") is not True
    ]
    if failure_steps:
        candidates.append(("post_tool_failure", max(failure_steps)))
    candidates.append(("latest_viable", None))
    repairs = []
    for seam_type, prefix_count in candidates:
        repaired = (
            _repair_trial(task, actions)
            if prefix_count is None
            else _repair_at_prefix(task, actions, prefix_count)
        )
        if repaired is not None:
            repairs.append((seam_type, *repaired))
    return repairs


def _heldout_entities(root: Path) -> tuple[set[str], set[str]]:
    instructions: set[str] = set()
    entities: set[str] = set()
    for split in ("dev", "test"):
        for task in load_tasks(root / f"{split}.jsonl"):
            instructions.add(task.instruction)
            target = task.goal.final_state.get("target")
            if isinstance(target, str):
                entities.add(target)
            presets = task.goal.final_state.get("presets", {})
            if isinstance(presets, dict):
                entities.update(str(name) for name in presets)
    return instructions, entities


def export_oec_sft(
    *,
    task_path: str | Path,
    task_selection_path: str | Path | None = None,
    student_run_path: str | Path,
    heldout_data_root: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
    seam_policy: str = "latest-v0.1",
) -> dict[str, Any]:
    task_source = Path(task_path)
    run_source = Path(student_run_path)
    heldout_root = Path(heldout_data_root)
    output = Path(output_path)
    manifest_destination = Path(manifest_path)
    audit_destination = Path(audit_path)
    for destination in (output, manifest_destination, audit_destination):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite OEC artifact: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(task_source)
    if any(task.split != "train" for task in tasks):
        raise ValueError("OEC accepts train-split tasks only")
    if task_selection_path is not None:
        tasks = load_rl_task_selection(
            selection_path=task_selection_path,
            source_task_path=task_source,
            tasks=tasks,
            split="train",
        )
    task_by_id = {task.id: task for task in tasks}
    run_header, trials = load_run(run_source)
    if run_header.get("split") != "train":
        raise ValueError("student run must use the train split")
    if run_header.get("task_suite_sha256") != hashlib.sha256(task_suite_payload(tasks)).hexdigest():
        raise ValueError("student run task suite does not match OEC tasks")
    if [trial["task_id"] for trial in trials] != [task.id for task in tasks]:
        raise ValueError("student run task order does not match OEC tasks")

    heldout_instructions, heldout_entities = _heldout_entities(heldout_root)
    if seam_policy not in {"latest-v0.1", "multi-v0.2"}:
        raise ValueError("unsupported OEC seam policy")
    generator = OEC_MULTI_FORMAT_VERSION if seam_policy == "multi-v0.2" else OEC_FORMAT_VERSION
    records: list[dict[str, Any]] = []
    seen_record_hashes: set[str] = set()
    audit_records: list[dict[str, Any]] = []
    rejected = 0
    skipped_success = 0
    for trial_data in trials:
        task = task_by_id[str(trial_data["task_id"])]
        if trial_data.get("grade", {}).get("success") is True:
            skipped_success += 1
            audit_records.append(
                {
                    "record_type": "repair",
                    "task_id": task.id,
                    "accepted": False,
                    "reason": "student_already_successful",
                }
            )
            continue
        repairs = (
            _multi_seam_repairs(task, trial_data)
            if seam_policy == "multi-v0.2"
            else [
                ("latest_viable", *repair)
                for repair in [_repair_trial(task, _student_actions(trial_data))]
                if repair is not None
            ]
        )
        if not repairs:
            rejected += 1
            audit_records.append(
                {
                    "record_type": "repair",
                    "task_id": task.id,
                    "accepted": False,
                    "reason": "no_verified_suffix_within_horizon",
                }
            )
            continue
        for seam_type, trial, prefix_count, expert_decisions in repairs:
            repair_id = f"{task.id}:{seam_type}:prefix-{prefix_count}"
            repair_record_hashes = []
            for expert_turn, (context, action) in enumerate(expert_decisions, 1):
                completion = _action_completion(action, call_id=f"oec_{expert_turn}")
                prompt = _context_messages(context)
                canonical = json.dumps(
                    {"prompt": prompt, "completion": [completion]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                canonical_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                repair_record_hashes.append(canonical_hash)
                if canonical_hash in seen_record_hashes:
                    continue
                seen_record_hashes.add(canonical_hash)
                records.append(
                    {
                        "prompt": prompt,
                        "completion": [completion],
                        "tools": tool_schemas(include_interaction=True),
                        "metadata": {
                            "generator": generator,
                            "source_split": "train",
                            "task_id": task.id,
                            "repair_id": repair_id,
                            "seam_type": seam_type,
                            "student_prefix_action_count": prefix_count,
                            "expert_turn": expert_turn,
                            "canonical_hash": canonical_hash,
                        },
                    }
                )
            prefix_events = [event for event in trial.events if event.step <= prefix_count]
            audit_records.append(
                {
                    "record_type": "repair",
                    "task_id": task.id,
                    "accepted": True,
                    "repair_id": repair_id,
                    "seam_type": seam_type,
                    "student_prefix_action_count": prefix_count,
                    "student_prefix_non_ok_tool_result_count": sum(
                        event.kind == "tool_result" and event.payload.get("ok") is not True
                        for event in prefix_events
                    ),
                    "expert_decision_count": len(expert_decisions),
                    "record_hashes": repair_record_hashes,
                    "repaired_trial": _replayable_trial_payload(trial),
                }
            )

    dataset_payload = _jsonl_payload(records)
    output.write_bytes(dataset_payload)
    audit_header = {
        "record_type": "oec_audit_header",
        "format_version": OEC_AUDIT_FORMAT_VERSION,
        "generator": generator,
        "seam_policy": seam_policy,
        "task_sha256": _sha256(task_source),
        "task_selection_path": (
            Path(task_selection_path).as_posix() if task_selection_path is not None else None
        ),
        "task_selection_sha256": (
            _sha256(Path(task_selection_path)) if task_selection_path is not None else None
        ),
        "student_run_sha256": _sha256(run_source),
        "record_count": len(audit_records),
    }
    audit_payload = _jsonl_payload([audit_header, *audit_records])
    audit_destination.write_bytes(audit_payload)
    completion_text = json.dumps(
        [record["completion"] for record in records], ensure_ascii=False
    )
    prompt_instructions = {
        str(record["prompt"][1]["content"]) for record in records if len(record["prompt"]) > 1
    }
    manifest = {
        "format_version": OEC_MANIFEST_FORMAT_VERSION,
        "generator": generator,
        "seam_policy": seam_policy,
        "task_path": task_source.as_posix(),
        "task_sha256": _sha256(task_source),
        "task_selection_path": (
            Path(task_selection_path).as_posix() if task_selection_path is not None else None
        ),
        "task_selection_sha256": (
            _sha256(Path(task_selection_path)) if task_selection_path is not None else None
        ),
        "student_run_path": run_source.as_posix(),
        "student_run_sha256": _sha256(run_source),
        "dataset_path": output.as_posix(),
        "dataset_sha256": hashlib.sha256(dataset_payload).hexdigest(),
        "audit_path": audit_destination.as_posix(),
        "audit_sha256": hashlib.sha256(audit_payload).hexdigest(),
        "record_count": len(records),
        "accepted_repair_count": sum(row.get("accepted") is True for row in audit_records),
        "rejected_repair_count": rejected,
        "skipped_success_count": skipped_success,
        "heldout_instruction_overlap_count": len(prompt_instructions & heldout_instructions),
        "heldout_completion_entities": sorted(
            entity for entity in heldout_entities if entity and entity in completion_text
        ),
        "decision_type_counts": dict(
            sorted(
                Counter(
                    record["completion"][0].get("tool_calls", [{}])[0]
                    .get("function", {})
                    .get("name", "message")
                    for record in records
                ).items()
            )
        ),
        "seam_type_counts": dict(
            sorted(
                Counter(
                    row.get("seam_type")
                    for row in audit_records
                    if row.get("accepted") is True
                ).items()
            )
        ),
    }
    manifest_destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_oec_sft(
    *,
    task_path: str | Path,
    task_selection_path: str | Path | None = None,
    student_run_path: str | Path,
    heldout_data_root: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
) -> dict[str, Any]:
    task_source = Path(task_path)
    run_source = Path(student_run_path)
    dataset_source = Path(dataset_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    audit_source = Path(audit_path)
    if manifest.get("format_version") != OEC_MANIFEST_FORMAT_VERSION:
        raise ValueError("unsupported OEC manifest format")
    for label, path in (
        ("task", task_source),
        ("student run", run_source),
        ("dataset", dataset_source),
        ("audit", audit_source),
    ):
        if manifest.get(f"{label.replace(' ', '_')}_sha256") != _sha256(path):
            raise ValueError(f"{label} SHA-256 mismatch")
    tasks = load_tasks(task_source)
    if task_selection_path is None and manifest.get("task_selection_sha256") is not None:
        raise ValueError("task selection is required by the OEC manifest")
    if task_selection_path is not None:
        selection_source = Path(task_selection_path)
        if manifest.get("task_selection_sha256") != _sha256(selection_source):
            raise ValueError("task selection SHA-256 mismatch")
        tasks = load_rl_task_selection(
            selection_path=selection_source,
            source_task_path=task_source,
            tasks=tasks,
            split="train",
        )
    task_by_id = {task.id: task for task in tasks}
    run_header, source_trials = load_run(run_source)
    if run_header.get("split") != "train" or any(task.split != "train" for task in tasks):
        raise ValueError("OEC evidence is not train-only")
    source_trial_by_id = {str(trial["task_id"]): trial for trial in source_trials}
    dataset_records = [
        json.loads(line) for line in dataset_source.read_text(encoding="utf-8").splitlines()
    ]
    audit_rows = [
        json.loads(line) for line in audit_source.read_text(encoding="utf-8").splitlines()
    ]
    if not audit_rows or audit_rows[0].get("format_version") != OEC_AUDIT_FORMAT_VERSION:
        raise ValueError("unsupported OEC audit format")
    for field in ("task_sha256", "task_selection_sha256", "student_run_sha256"):
        if audit_rows[0].get(field) != manifest.get(field):
            raise ValueError(f"OEC audit {field} mismatch")
    if audit_rows[0].get("record_count") != len(audit_rows) - 1:
        raise ValueError("OEC audit count mismatch")
    record_by_hash = {
        str(record.get("metadata", {}).get("canonical_hash")): record
        for record in dataset_records
    }
    for record in dataset_records:
        canonical = json.dumps(
            {
                "prompt": record.get("prompt"),
                "completion": record.get("completion"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if actual_hash != record.get("metadata", {}).get("canonical_hash"):
            raise ValueError("OEC dataset canonical hash mismatch")
    if len(record_by_hash) != len(dataset_records):
        raise ValueError("OEC dataset contains duplicate canonical hashes")
    replayed = 0
    for repair in audit_rows[1:]:
        if repair.get("accepted") is not True:
            continue
        task_id = str(repair["task_id"])
        if task_id not in task_by_id or task_id not in source_trial_by_id:
            raise ValueError("OEC repair references an unknown task")
        task = task_by_id[task_id]
        prefix_count = int(repair["student_prefix_action_count"])
        source_actions = _student_actions(source_trial_by_id[task_id])
        regenerated = _repair_at_prefix(task, source_actions, prefix_count)
        if regenerated is None or regenerated[1] != prefix_count:
            raise ValueError("OEC student prefix no longer replays")
        replayed_trial, _, expert_decisions = regenerated
        stored_trial = repair["repaired_trial"]
        if _replayable_trial_payload(replayed_trial) != stored_trial:
            raise ValueError("OEC repaired trial does not replay")
        recomputed_grade = grade_task(
            goal=task.goal,
            final_state=replayed_trial.final_state,
            events=replayed_trial.events,
        )
        if not recomputed_grade.success:
            raise ValueError("OEC expert suffix is not successful")
        expected_hashes = []
        for turn, (context, action) in enumerate(expert_decisions, 1):
            completion = _action_completion(action, call_id=f"oec_{turn}")
            canonical = json.dumps(
                {"prompt": _context_messages(context), "completion": [completion]},
                ensure_ascii=False,
                sort_keys=True,
            )
            expected_hashes.append(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
        if expected_hashes != repair.get("record_hashes"):
            raise ValueError("OEC repair record hashes do not replay")
        if any(record_hash not in record_by_hash for record_hash in expected_hashes):
            raise ValueError("OEC dataset is missing a verified repair decision")
        replayed += 1
    heldout_instructions, heldout_entities = _heldout_entities(Path(heldout_data_root))
    prompt_instructions = {
        str(record["prompt"][1]["content"])
        for record in dataset_records
        if len(record.get("prompt", [])) > 1
    }
    completion_text = json.dumps(
        [record.get("completion") for record in dataset_records], ensure_ascii=False
    )
    entity_overlap = {entity for entity in heldout_entities if entity in completion_text}
    instruction_overlap = prompt_instructions & heldout_instructions
    if instruction_overlap or entity_overlap:
        raise ValueError("OEC dataset leaks held-out evaluation content")
    if replayed != manifest.get("accepted_repair_count"):
        raise ValueError("OEC accepted repair count mismatch")
    if len(dataset_records) != manifest.get("record_count"):
        raise ValueError("OEC dataset record count mismatch")
    return {
        "status": "verified",
        "format_version": OEC_MANIFEST_FORMAT_VERSION,
        "replayed_repair_count": replayed,
        "record_count": len(dataset_records),
        "heldout_instruction_overlap_count": len(instruction_overlap),
        "heldout_completion_entity_overlap_count": len(entity_overlap),
    }
