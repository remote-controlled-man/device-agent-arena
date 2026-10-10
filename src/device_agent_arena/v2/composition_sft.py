from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from device_agent_arena.domain.models import ToolCall
from device_agent_arena.v1.sft import (
    _assistant_message,
    _tool_schema,
)
from device_agent_arena.v2.composition_agents import (
    OracleCompositionAgent,
    _resolve_placeholders,
)
from device_agent_arena.v2.composition_environment import CompositionEnvironment
from device_agent_arena.v2.composition_io import load_composition_tasks
from device_agent_arena.v2.composition_models import V2CompositionTask
from device_agent_arena.v2.composition_suite import validate_composition_suite
from device_agent_arena.v2.protocol import (
    composition_messages_for_observation,
    functiongemma_record,
)


def _record_for_composition_action(
    *,
    task: V2CompositionTask,
    observation: Any,
    action: ToolCall,
    metadata: dict[str, Any],
    protocol: str = "functiongemma",
) -> dict[str, Any]:
    call_index = (
        sum(event.kind == "tool_call" for event in observation.events) + 1
    )
    record = {
        "prompt": composition_messages_for_observation(observation),
        "completion": [
            _assistant_message(action, call_id=f"call_{call_index}")
        ],
        "tools": [_tool_schema(tool) for tool in observation.tools],
        "metadata": {
            "task_id": task.id,
            "split": task.split,
            "slice_id": task.slice_id,
            "domain": task.domain,
            "variant": task.provenance.get("variant", "unknown"),
            "decision_kind": action.name,
            "format_version": "deviceops-ul-v2-composition-sft-v0.1",
            **metadata,
        },
    }
    return _format_protocol_record(record, protocol)


def _format_protocol_record(record: dict[str, Any], protocol: str) -> dict[str, Any]:
    if protocol == "functiongemma":
        return functiongemma_record(record)
    if protocol == "qwen":
        record["metadata"] = dict(record["metadata"])
        record["metadata"]["protocol"] = "qwen-tool-native-v2-composition"
        return record
    raise ValueError(f"unsupported composition SFT protocol: {protocol}")


def _records_for_composition_task(
    task: V2CompositionTask, *, protocol: str = "functiongemma"
) -> tuple[dict[str, Any], ...]:
    if task.split != "train":
        raise ValueError("composition SFT export is Train-only")
    environment = CompositionEnvironment()
    observation = environment.reset(task)
    agent = OracleCompositionAgent(task)
    tools = [_tool_schema(tool) for tool in observation.tools]
    records = []
    call_index = 0
    decision_index = 0
    while True:
        action = agent.act(observation)
        if not isinstance(action, ToolCall):
            raise TypeError("composition oracle SFT expects native interaction tool calls")
        decision_index += 1
        call_index += 1
        record = {
            "prompt": composition_messages_for_observation(observation),
            "completion": [
                _assistant_message(action, call_id=f"call_{call_index}")
            ],
            "tools": deepcopy(tools),
            "metadata": {
                "task_id": task.id,
                "split": task.split,
                "slice_id": task.slice_id,
                "domain": task.domain,
                "variant": task.provenance.get("variant", "unknown"),
                "decision_index": decision_index,
                "decision_kind": action.name,
                "format_version": "deviceops-ul-v2-composition-sft-v0.1",
            },
        }
        records.append(_format_protocol_record(record, protocol))
        transition = environment.step(action)
        if transition.done:
            trial = environment.finish(agent.name)
            if not trial.grade.success:
                raise RuntimeError(
                    f"composition oracle SFT trajectory failed grading: {task.id}"
                )
            break
        observation = transition.observation
    if decision_index != len(task.oracle_plan):
        raise RuntimeError(
            f"composition oracle decision count mismatch: {task.id} "
            f"({decision_index} != {len(task.oracle_plan)})"
        )
    return tuple(records)


def _plan_action(task: V2CompositionTask, index: int) -> ToolCall:
    step = task.oracle_plan[index]
    if step.kind != "tool":
        raise TypeError("composition recovery curriculum requires a tool decision")
    if any(value == "$last_operation" for value in step.arguments.values()):
        raise ValueError("recovery correction cannot depend on an unresolved operation")
    return ToolCall(step.name, deepcopy(step.arguments))


def _target_action(task: V2CompositionTask) -> ToolCall:
    target_id = str(task.public_context["target_id"])
    spec = next(item for item in task.action_specs if item.target_id == target_id)
    return ToolCall(
        "apply_action",
        {"action": spec.action, "target_id": target_id, "payload": {}},
    )


def _recovery_records_for_composition_task(
    task: V2CompositionTask,
    *,
    protocol: str = "functiongemma",
) -> tuple[dict[str, Any], ...]:
    """Create public-state corrections after plausible model deviations.

    Labels come from the Train oracle, while every prompt contains only public messages,
    tool schemas, calls, and results. No hidden grader state is serialized.
    """
    if task.split != "train":
        raise ValueError("composition recovery SFT export is Train-only")
    records: list[dict[str, Any]] = []
    environment = CompositionEnvironment()
    environment.reset(task)
    for index in range(2):
        transition = environment.step(_plan_action(task, index))
        if transition.done:
            raise RuntimeError("composition recovery prefix terminated unexpectedly")

    wrong_action = _target_action(task)
    correction = _plan_action(task, 2)
    first = environment.step(wrong_action)
    if not first.done and first.tool_result is not None and not first.tool_result.ok:
        records.append(
            _record_for_composition_action(
                task=task,
                observation=first.observation,
                action=correction,
                metadata={
                    "curriculum_source": "composition-recovery",
                    "recovery_kind": "after_first_precondition_failure",
                },
                protocol=protocol,
            )
        )
        second = environment.step(wrong_action)
        if not second.done:
            records.append(
                _record_for_composition_action(
                    task=task,
                    observation=second.observation,
                    action=correction,
                    metadata={
                        "curriculum_source": "composition-recovery",
                        "recovery_kind": "after_repeated_precondition_failure",
                    },
                    protocol=protocol,
                )
            )

    if task.provenance.get("variant") == "infeasible":
        environment = CompositionEnvironment()
        environment.reset(task)
        for index in range(2):
            environment.step(_plan_action(task, index))
        relation = next(
            item
            for item in task.initial_world["relations"]
            if item.get("user_action_key")
        )
        unavailable = environment.step(
            ToolCall(
                "request_user_action",
                {
                    "action_key": str(relation["user_action_key"]),
                    "message": "请完成公开依赖中列出的用户侧动作。",
                },
            )
        )
        if unavailable.done or unavailable.tool_result is None:
            raise RuntimeError("infeasible recovery prefix terminated unexpectedly")
        if unavailable.tool_result.code != "user_action_unavailable":
            raise RuntimeError("expected unavailable user-side action evidence")
        records.append(
            _record_for_composition_action(
                task=task,
                observation=unavailable.observation,
                action=correction,
                metadata={
                    "curriculum_source": "composition-recovery",
                    "recovery_kind": "after_user_action_unavailable",
                },
                protocol=protocol,
            )
        )
    return tuple(records)


def _read_replay(path: Path, modulus: int) -> list[dict[str, Any]]:
    if modulus < 1:
        raise ValueError("replay modulus must be positive")
    selected = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or index % modulus:
            continue
        record = json.loads(line)
        task_id = str(record.get("metadata", {}).get("task_id", ""))
        if "-dev-" in task_id or "-test-" in task_id:
            raise ValueError("composition replay must remain Train-only")
        item = deepcopy(record)
        item["metadata"] = dict(item.get("metadata", {}))
        item["metadata"]["curriculum_source"] = "v2.3-replay"
        selected.append(item)
    return selected


def export_functiongemma_composition_sft(
    *,
    suite_root: str | Path,
    output: str | Path,
    manifest_output: str | Path,
    composition_repeats: int = 2,
    replay_data: str | Path | None = None,
    replay_modulus: int = 8,
) -> dict[str, Any]:
    if composition_repeats < 1:
        raise ValueError("composition repeats must be positive")
    root = Path(suite_root)
    validate_composition_suite(root)
    tasks = load_composition_tasks(root / "train.jsonl")
    composition = [
        record for task in tasks for record in _records_for_composition_task(task)
    ]
    records: list[dict[str, Any]] = []
    for repeat_index in range(composition_repeats):
        for source in composition:
            item = deepcopy(source)
            item["metadata"] = dict(item["metadata"])
            item["metadata"]["curriculum_source"] = "composition-oracle"
            item["metadata"]["repeat_index"] = repeat_index
            records.append(item)
    replay = []
    if replay_data is not None:
        replay = _read_replay(Path(replay_data), replay_modulus)
        records.extend(replay)

    destination = Path(output)
    manifest_path = Path(manifest_output)
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite composition SFT artifacts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    decision_counts = Counter(
        str(record["metadata"].get("decision_kind", "unknown")) for record in records
    )
    source_counts = Counter(
        str(record["metadata"].get("curriculum_source", "unknown"))
        for record in records
    )
    data_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "format_version": "deviceops-ul-v2-composition-sft-manifest-v0.1",
        "record_count": len(records),
        "composition_source_record_count": len(composition),
        "composition_repeats": composition_repeats,
        "replay_record_count": len(replay),
        "replay_modulus": replay_modulus if replay_data is not None else None,
        "decision_counts": dict(sorted(decision_counts.items())),
        "curriculum_source_counts": dict(sorted(source_counts.items())),
        "sha256": data_sha256,
        "data_sha256": data_sha256,
        "suite_manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "test_data_used": False,
        "dev_data_used": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def export_functiongemma_composition_recovery_sft(
    *,
    suite_root: str | Path,
    output: str | Path,
    manifest_output: str | Path,
    hard_state_repeats: int = 4,
    replay_data: str | Path | None = None,
    replay_modulus: int = 8,
) -> dict[str, Any]:
    if hard_state_repeats < 1:
        raise ValueError("hard-state repeats must be positive")
    root = Path(suite_root)
    validate_composition_suite(root)
    tasks = load_composition_tasks(root / "train.jsonl")
    successful_hard_states = [
        record
        for task in tasks
        for record in _records_for_composition_task(task)
        if int(record["metadata"]["decision_index"]) >= 3
    ]
    recovery_states = [
        record
        for task in tasks
        for record in _recovery_records_for_composition_task(task)
    ]
    hard_states = successful_hard_states + recovery_states
    records: list[dict[str, Any]] = []
    for repeat_index in range(hard_state_repeats):
        for source in hard_states:
            item = deepcopy(source)
            item["metadata"] = dict(item["metadata"])
            item["metadata"].setdefault(
                "curriculum_source", "composition-hard-success"
            )
            item["metadata"]["repeat_index"] = repeat_index
            records.append(item)
    replay = []
    if replay_data is not None:
        replay = _read_replay(Path(replay_data), replay_modulus)
        records.extend(replay)

    destination = Path(output)
    manifest_path = Path(manifest_output)
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite composition recovery SFT artifacts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    source_counts = Counter(
        str(record["metadata"].get("curriculum_source", "unknown"))
        for record in records
    )
    recovery_counts = Counter(
        str(record["metadata"].get("recovery_kind", "none"))
        for record in records
        if record["metadata"].get("curriculum_source") == "composition-recovery"
    )
    manifest = {
        "format_version": "deviceops-ul-v2-composition-recovery-sft-manifest-v0.1",
        "record_count": len(records),
        "successful_hard_state_source_count": len(successful_hard_states),
        "recovery_state_source_count": len(recovery_states),
        "hard_state_repeats": hard_state_repeats,
        "replay_record_count": len(replay),
        "replay_modulus": replay_modulus if replay_data is not None else None,
        "curriculum_source_counts": dict(sorted(source_counts.items())),
        "recovery_kind_counts": dict(sorted(recovery_counts.items())),
        "sha256": digest,
        "data_sha256": digest,
        "suite_manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "test_data_used": False,
        "dev_data_used": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def export_qwen_composition_curriculum_sft(
    *,
    suite_root: str | Path,
    output: str | Path,
    manifest_output: str | Path,
    oracle_repeats: int = 2,
    recovery_repeats: int = 4,
) -> dict[str, Any]:
    """Export Train-only Qwen next-decision data using the exact runtime prompt."""
    if oracle_repeats < 1 or recovery_repeats < 1:
        raise ValueError("Qwen composition repeat counts must be positive")
    root = Path(suite_root)
    validate_composition_suite(root)
    tasks = load_composition_tasks(root / "train.jsonl")
    oracle = [
        record
        for task in tasks
        for record in _records_for_composition_task(task, protocol="qwen")
    ]
    recovery = [
        record
        for task in tasks
        for record in _recovery_records_for_composition_task(task, protocol="qwen")
    ]
    records: list[dict[str, Any]] = []
    for repeat_index in range(oracle_repeats):
        for source in oracle:
            item = deepcopy(source)
            item["metadata"] = dict(item["metadata"])
            item["metadata"]["curriculum_source"] = "composition-oracle"
            item["metadata"]["repeat_index"] = repeat_index
            records.append(item)
    for repeat_index in range(recovery_repeats):
        for source in recovery:
            item = deepcopy(source)
            item["metadata"] = dict(item["metadata"])
            item["metadata"]["curriculum_source"] = "composition-recovery"
            item["metadata"]["repeat_index"] = repeat_index
            records.append(item)

    destination = Path(output)
    manifest_path = Path(manifest_output)
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite Qwen composition SFT artifacts")
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    hidden_markers = {
        str(task.initial_world["secret_marker"]).encode("utf-8") for task in tasks
    }
    if any(marker in serialized for marker in hidden_markers):
        raise ValueError("hidden composition world marker leaked into Qwen SFT data")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(serialized)
    digest = hashlib.sha256(serialized).hexdigest()
    decision_counts = Counter(
        str(record["metadata"].get("decision_kind", "unknown"))
        for record in records
    )
    source_counts = Counter(
        str(record["metadata"].get("curriculum_source", "unknown"))
        for record in records
    )
    manifest = {
        "format_version": "deviceops-ul-v2-qwen-composition-sft-manifest-v0.1",
        "protocol": "qwen-tool-native-v2-composition",
        "record_count": len(records),
        "oracle_source_record_count": len(oracle),
        "recovery_source_record_count": len(recovery),
        "oracle_repeats": oracle_repeats,
        "recovery_repeats": recovery_repeats,
        "decision_counts": dict(sorted(decision_counts.items())),
        "curriculum_source_counts": dict(sorted(source_counts.items())),
        "sha256": digest,
        "data_sha256": digest,
        "suite_manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "test_policy_amendment_sha256": (
            hashlib.sha256(
                (root / "manifest.test-policy-v0.2.json").read_bytes()
            ).hexdigest()
            if (root / "manifest.test-policy-v0.2.json").is_file()
            else None
        ),
        "hidden_marker_count": len(hidden_markers),
        "hidden_marker_leaks": 0,
        "dev_data_used": False,
        "test_data_used": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _qwen_failure_corrections(
    task: V2CompositionTask,
    trial: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Replay one Train trial and label each observed oracle deviation."""
    environment = CompositionEnvironment()
    observation = environment.reset(task)
    oracle_index = 0
    corrections: list[dict[str, Any]] = []
    events_by_step: dict[int, list[dict[str, Any]]] = {}
    for event in trial.get("events", []):
        events_by_step.setdefault(int(event["step"]), []).append(event)

    for step_number in sorted(events_by_step):
        calls = [
            event
            for event in events_by_step[step_number]
            if event.get("kind") == "tool_call"
        ]
        if len(calls) != 1:
            raise ValueError(
                f"failure trial {task.id} step {step_number} must contain one tool call"
            )
        payload = calls[0].get("payload", {})
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("failure trial tool arguments must be an object")
        actual = ToolCall(str(payload.get("name", "")), deepcopy(arguments))
        if oracle_index >= len(task.oracle_plan):
            raise ValueError(f"failure trial {task.id} continues beyond the oracle plan")
        oracle_step = task.oracle_plan[oracle_index]
        if oracle_step.kind != "tool":
            raise TypeError("Qwen failure refinement requires tool-native oracle steps")
        expected = ToolCall(
            oracle_step.name,
            _resolve_placeholders(oracle_step.arguments, observation.events),
        )
        if actual != expected:
            corrections.append(
                _record_for_composition_action(
                    task=task,
                    observation=observation,
                    action=expected,
                    metadata={
                        "curriculum_source": "train-failure-correction",
                        "oracle_decision_index": oracle_index + 1,
                        "observed_wrong_action": actual.name,
                        "observed_wrong_step": step_number,
                    },
                    protocol="qwen",
                )
            )
        else:
            oracle_index += 1
        transition = environment.step(actual)
        if transition.done:
            break
        observation = transition.observation
    return tuple(corrections)


def _prompt_conflict_count(records: list[dict[str, Any]]) -> int:
    completions_by_prompt: dict[str, set[str]] = {}
    for record in records:
        prompt_key = json.dumps(
            [record["prompt"], record["tools"]],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        completion = json.dumps(
            record["completion"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        completions_by_prompt.setdefault(prompt_key, set()).add(completion)
    return sum(len(labels) > 1 for labels in completions_by_prompt.values())


def export_qwen_failure_refinement_sft(
    *,
    suite_root: str | Path,
    failure_run: str | Path,
    replay_data: str | Path,
    output: str | Path,
    manifest_output: str | Path,
    correction_repeats: int = 8,
    hard_oracle_repeats: int = 3,
    recovery_repeats: int = 2,
    replay_modulus: int = 4,
) -> dict[str, Any]:
    """Build Train-only hard-example refinement from replayed model failures."""
    repeat_counts = (correction_repeats, hard_oracle_repeats, recovery_repeats)
    if any(value < 1 for value in repeat_counts):
        raise ValueError("Qwen refinement repeat counts must be positive")
    root = Path(suite_root)
    validate_composition_suite(root)
    tasks = load_composition_tasks(root / "train.jsonl")
    task_by_id = {task.id: task for task in tasks}

    run_path = Path(failure_run)
    run_bytes = run_path.read_bytes()
    run_records = [
        json.loads(line)
        for line in run_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not run_records or run_records[0].get("record_type") != "run_header":
        raise ValueError("failure refinement requires a composition run header")
    header = run_records[0]
    if header.get("split") != "train":
        raise ValueError("failure refinement is Train-only")
    trials = [record for record in run_records[1:] if record.get("record_type") == "trial"]
    if len(trials) != header.get("task_count"):
        raise ValueError("failure run task count does not match its header")
    trial_ids = [str(trial.get("task_id", "")) for trial in trials]
    has_unknown_task = any(task_id not in task_by_id for task_id in trial_ids)
    if len(set(trial_ids)) != len(trial_ids) or has_unknown_task:
        raise ValueError("failure run has duplicate or non-Train task IDs")
    task_ids_sha256 = hashlib.sha256(
        ("\n".join(trial_ids) + "\n").encode("utf-8")
    ).hexdigest()
    if task_ids_sha256 != header.get("task_ids_sha256"):
        raise ValueError("failure run task IDs do not match its header")
    selected_payload = "".join(
        json.dumps(asdict(task_by_id[task_id]), ensure_ascii=False, sort_keys=True) + "\n"
        for task_id in trial_ids
    ).encode("utf-8")
    if hashlib.sha256(selected_payload).hexdigest() != header.get("task_suite_sha256"):
        raise ValueError("failure run task payload does not match the frozen Train suite")

    failed_trials = [trial for trial in trials if not trial.get("grade", {}).get("success")]
    if not failed_trials:
        raise ValueError("failure refinement run contains no failed Train tasks")
    failed_tasks = [task_by_id[str(trial["task_id"])] for trial in failed_trials]
    corrections = [
        record
        for task, trial in zip(failed_tasks, failed_trials, strict=True)
        for record in _qwen_failure_corrections(task, trial)
    ]
    if not corrections:
        raise ValueError("failed Train trials yielded no oracle deviations")
    hard_oracle = [
        record
        for task in failed_tasks
        for record in _records_for_composition_task(task, protocol="qwen")
    ]
    recovery = [
        record
        for task in failed_tasks
        for record in _recovery_records_for_composition_task(task, protocol="qwen")
    ]
    replay = _read_replay(Path(replay_data), replay_modulus)
    for record in replay:
        record["metadata"]["curriculum_source"] = "qwen-v2.7-replay"

    records: list[dict[str, Any]] = []
    for sources, repeats, source_name in (
        (corrections, correction_repeats, "train-failure-correction"),
        (hard_oracle, hard_oracle_repeats, "failed-task-oracle"),
        (recovery, recovery_repeats, "failed-task-recovery"),
    ):
        for repeat_index in range(repeats):
            for source in sources:
                item = deepcopy(source)
                item["metadata"] = dict(item["metadata"])
                item["metadata"]["curriculum_source"] = source_name
                item["metadata"]["repeat_index"] = repeat_index
                records.append(item)
    records.extend(replay)

    destination = Path(output)
    manifest_path = Path(manifest_output)
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite Qwen refinement artifacts")
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    hidden_markers = {
        str(task.initial_world["secret_marker"]).encode("utf-8") for task in tasks
    }
    if any(marker in serialized for marker in hidden_markers):
        raise ValueError("hidden composition world marker leaked into Qwen refinement data")
    prompt_conflicts = _prompt_conflict_count(records)
    if prompt_conflicts:
        raise ValueError("Qwen refinement data contains conflicting prompt labels")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(serialized)
    digest = hashlib.sha256(serialized).hexdigest()
    source_counts = Counter(
        str(record["metadata"].get("curriculum_source", "unknown"))
        for record in records
    )
    decision_counts = Counter(
        str(record["metadata"].get("decision_kind", "unknown"))
        for record in records
    )
    manifest = {
        "format_version": "deviceops-ul-v2-qwen-failure-refinement-manifest-v0.1",
        "protocol": "qwen-tool-native-v2-composition",
        "record_count": len(records),
        "failed_task_count": len(failed_tasks),
        "failed_task_ids": [task.id for task in failed_tasks],
        "correction_source_record_count": len(corrections),
        "hard_oracle_source_record_count": len(hard_oracle),
        "recovery_source_record_count": len(recovery),
        "replay_record_count": len(replay),
        "correction_repeats": correction_repeats,
        "hard_oracle_repeats": hard_oracle_repeats,
        "recovery_repeats": recovery_repeats,
        "replay_modulus": replay_modulus,
        "curriculum_source_counts": dict(sorted(source_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "failure_run_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "sha256": digest,
        "data_sha256": digest,
        "suite_manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "prompt_conflict_count": prompt_conflicts,
        "hidden_marker_count": len(hidden_markers),
        "hidden_marker_leaks": 0,
        "dev_data_used": False,
        "test_data_used": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
