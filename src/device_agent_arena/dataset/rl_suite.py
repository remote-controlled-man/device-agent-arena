from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from device_agent_arena.dataset.generate_rl import (
    Split,
    generate_rl_tasks,
    generate_rl_tasks_v2,
    generate_rl_tasks_v3,
)
from device_agent_arena.domain.models import TaskSpec
from device_agent_arena.domain.task_io import load_tasks, write_tasks

RL_SUITE_FORMAT_VERSION = "rl-task-suite-v0.1"
RL_TASK_SELECTION_FORMAT_VERSION = "rl-task-selection-v0.1"
RL_SUITE_FORMAT_VERSIONS = {
    "v0.1": RL_SUITE_FORMAT_VERSION,
    "v0.2": "rl-task-suite-v0.2",
    "v0.3": "rl-task-suite-v0.3",
}
RL_SPLITS = ("train", "dev", "test")
REQUIRED_LONG_HORIZON_TOOLS = {
    "connect_device",
    "calibrate_device",
    "set_mode",
    "set_orientation",
    "set_tracking",
    "save_preset",
}


def write_rl_task_suite(
    output_root: str | Path,
    *,
    split_counts: Mapping[Split, int] | None = None,
    suite_version: str = "v0.1",
    root_seed: int = 0,
) -> dict[str, Any]:
    root = Path(output_root)
    counts = dict(split_counts or {"train": 128, "dev": 32, "test": 64})
    if set(counts) != set(RL_SPLITS):
        raise ValueError(f"split_counts must contain exactly {RL_SPLITS}")
    if suite_version not in RL_SUITE_FORMAT_VERSIONS:
        raise ValueError("suite_version must be v0.1, v0.2, or v0.3")
    if not isinstance(root_seed, int) or isinstance(root_seed, bool) or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    if suite_version != "v0.3" and root_seed:
        raise ValueError("root_seed is supported only by suite_version v0.3")
    generators = {
        "v0.1": generate_rl_tasks,
        "v0.2": generate_rl_tasks_v2,
        "v0.3": generate_rl_tasks_v3,
    }
    generator = generators[suite_version]

    files: dict[str, dict[str, Any]] = {}
    generated: dict[str, tuple[TaskSpec, ...]] = {}
    for split in RL_SPLITS:
        tasks = (
            generator(split, count=counts[split], root_seed=root_seed)
            if suite_version == "v0.3"
            else generator(split, count=counts[split])
        )
        generated[split] = tasks
        filename = f"{split}.jsonl"
        path = root / filename
        write_tasks(path, list(tasks))
        payload = path.read_bytes()
        files[filename] = {
            "record_count": len(tasks),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "format_version": RL_SUITE_FORMAT_VERSIONS[suite_version],
        "generator": f"rl-builtin-{suite_version}",
        "files": files,
    }
    if suite_version == "v0.3":
        manifest.update(
            {
                "root_seed": root_seed,
                "seed_scheme": "split-base-plus-root-v0.1",
                "structure_catalogue_version": "v0.1",
                "split_statistics": {
                    split: _v3_split_statistics(generated[split]) for split in RL_SPLITS
                },
            }
        )
    (root / "manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    validate_rl_task_suite(root)
    return manifest


def validate_rl_task_suite(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    format_version = manifest.get("format_version")
    if format_version not in set(RL_SUITE_FORMAT_VERSIONS.values()):
        raise ValueError("unsupported RL task suite format_version")
    files = manifest.get("files")
    expected_files = {f"{split}.jsonl" for split in RL_SPLITS}
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError("RL task suite manifest must describe train/dev/test JSONL")

    all_tasks: list[TaskSpec] = []
    tasks_by_split: dict[str, tuple[TaskSpec, ...]] = {}
    split_counts: dict[str, int] = {}
    for split in RL_SPLITS:
        filename = f"{split}.jsonl"
        path = root / filename
        payload = path.read_bytes()
        metadata = files[filename]
        if metadata.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"{filename} SHA-256 does not match manifest")
        tasks = load_tasks(path)
        if metadata.get("record_count") != len(tasks):
            raise ValueError(f"{filename} record count does not match manifest")
        if any(task.split != split for task in tasks):
            raise ValueError(f"{filename} contains a task from another split")
        if format_version == RL_SUITE_FORMAT_VERSIONS["v0.3"]:
            _validate_v3_contract(tasks)
        else:
            _validate_long_horizon_contract(tasks)
        if format_version == RL_SUITE_FORMAT_VERSIONS["v0.2"]:
            _validate_v2_observability(tasks)
        all_tasks.extend(tasks)
        tasks_by_split[split] = tasks
        split_counts[split] = len(tasks)

    ids = [task.id for task in all_tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task id across RL splits")
    _validate_split_disjoint_entities(all_tasks)
    if format_version == RL_SUITE_FORMAT_VERSIONS["v0.3"]:
        _validate_v3_structural_holdouts(all_tasks)
        _validate_v3_atomic_factor_coverage(all_tasks)
        _validate_v3_manifest_metadata(manifest, tasks_by_split)
    return {
        "valid": True,
        "format_version": format_version,
        "split_counts": dict(sorted(split_counts.items())),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def write_rl_task_selection(
    *,
    source_task_path: str | Path,
    output_path: str | Path,
    split: Split,
    task_ids: tuple[str, ...] | list[str],
    selection_policy: str,
) -> dict[str, Any]:
    source = Path(source_task_path)
    tasks = load_tasks(source)
    manifest, _ = _build_rl_task_selection(
        tasks=tasks,
        source_task_path=source,
        split=split,
        task_ids=tuple(task_ids),
        selection_policy=selection_policy,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    return manifest


def load_rl_task_selection(
    *,
    selection_path: str | Path,
    source_task_path: str | Path,
    tasks: tuple[TaskSpec, ...],
    split: Split,
) -> tuple[TaskSpec, ...]:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if selection.get("format_version") != RL_TASK_SELECTION_FORMAT_VERSION:
        raise ValueError("unsupported RL task selection format_version")
    if selection.get("split") != split:
        raise ValueError("RL task selection split does not match the requested split")
    task_ids = selection.get("task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
        raise ValueError("RL task selection task_ids must be a string list")
    task_ids_payload = _task_ids_payload(tuple(task_ids))
    if selection.get("task_ids_sha256") != hashlib.sha256(task_ids_payload).hexdigest():
        raise ValueError("RL task selection task ID hash does not match")
    rebuilt, selected = _build_rl_task_selection(
        tasks=tasks,
        source_task_path=Path(source_task_path),
        split=split,
        task_ids=tuple(task_ids),
        selection_policy=str(selection.get("selection_policy", "")),
    )
    if selection != rebuilt:
        raise ValueError("RL task selection metadata does not match the frozen task source")
    return selected


def _build_rl_task_selection(
    *,
    tasks: tuple[TaskSpec, ...],
    source_task_path: Path,
    split: Split,
    task_ids: tuple[str, ...],
    selection_policy: str,
) -> tuple[dict[str, Any], tuple[TaskSpec, ...]]:
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("RL task selection IDs must be non-empty and unique")
    if not selection_policy.strip():
        raise ValueError("RL task selection policy must be non-empty")
    by_id = {task.id: task for task in tasks}
    unknown = set(task_ids) - set(by_id)
    if unknown:
        raise ValueError(f"RL task selection contains unknown IDs: {sorted(unknown)}")
    selected = tuple(by_id[task_id] for task_id in task_ids)
    if any(task.split != split for task in selected):
        raise ValueError("RL task selection contains a task from another split")
    coverage = {
        "task_count": len(selected),
        "structure_counts": dict(
            sorted(
                Counter(
                    task.provenance.get("structure", "unknown") for task in selected
                ).items()
            )
        ),
        "workflow_counts": dict(
            sorted(Counter(task.provenance.get("workflow", "unknown") for task in selected).items())
        ),
        "initial_profile_counts": dict(
            sorted(
                Counter(
                    task.provenance.get("initial_profile", "unknown")
                    for task in selected
                ).items()
            )
        ),
        "fault_plan_count": sum(bool(task.fault_plan) for task in selected),
        "clarification_count": sum(bool(task.goal.required_clarifications) for task in selected),
    }
    task_ids_payload = _task_ids_payload(task_ids)
    manifest = {
        "format_version": RL_TASK_SELECTION_FORMAT_VERSION,
        "split": split,
        "source_task_sha256": hashlib.sha256(source_task_path.read_bytes()).hexdigest(),
        "task_ids": list(task_ids),
        "task_ids_sha256": hashlib.sha256(task_ids_payload).hexdigest(),
        "selection_policy": selection_policy.strip(),
        "coverage": coverage,
    }
    return manifest, selected


def _task_ids_payload(task_ids: tuple[str, ...]) -> bytes:
    return json.dumps(task_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _validate_long_horizon_contract(tasks: tuple[TaskSpec, ...]) -> None:
    for task in tasks:
        if set(task.goal.required_tools) != REQUIRED_LONG_HORIZON_TOOLS:
            raise ValueError(f"{task.id} does not require the six-step device workflow")
        if not {"rl_compositional", "rl_long_horizon"}.issubset(task.categories):
            raise ValueError(f"{task.id} is missing RL long-horizon categories")
        if task.goal.terminal_behavior != "answer":
            raise ValueError(f"{task.id} must terminate with an answer")


def _validate_split_disjoint_entities(tasks: list[TaskSpec]) -> None:
    targets: dict[str, set[str]] = {split: set() for split in RL_SPLITS}
    presets: dict[str, set[str]] = {split: set() for split in RL_SPLITS}
    for task in tasks:
        target = task.goal.final_state.get("target")
        if isinstance(target, str):
            targets[task.split].add(target)
        preset_data = task.goal.final_state.get("presets", {})
        if isinstance(preset_data, dict):
            presets[task.split].update(preset_data)
    for index, split in enumerate(RL_SPLITS):
        for other in RL_SPLITS[index + 1 :]:
            if targets[split] & targets[other] or presets[split] & presets[other]:
                raise ValueError(f"RL entities overlap between {split} and {other}")


def _validate_v2_observability(tasks: tuple[TaskSpec, ...]) -> None:
    for task in tasks:
        kind_word = "人物" if task.goal.final_state.get("tracking") == "person" else "物体"
        target = task.goal.final_state.get("target")
        if task.version != "1.1" or task.provenance.get("generator") != "rl-builtin-v0.2":
            raise ValueError(f"{task.id} does not carry the v0.2 task identity")
        if kind_word not in task.instruction:
            raise ValueError(f"{task.id} hides its required tracking kind")
        if task.goal.required_clarifications:
            if f"指定{kind_word}" not in task.instruction or target in task.instruction:
                raise ValueError(f"{task.id} has an invalid missing-target instruction")
            if task.clarifications.get("target") != target:
                raise ValueError(f"{task.id} clarification does not reveal the target")
        elif target not in task.instruction:
            raise ValueError(f"{task.id} hides an explicit target label")


def _validate_v3_contract(tasks: tuple[TaskSpec, ...]) -> None:
    structures: set[str] = set()
    required_sequences: set[tuple[str, ...]] = set()
    for task in tasks:
        structure = task.provenance.get("structure")
        if (
            task.version != "1.2"
            or task.provenance.get("generator") != "rl-builtin-v0.3"
            or not structure
        ):
            raise ValueError(f"{task.id} does not carry the v0.3 task identity")
        expected_structure = "|".join(
            (
                task.provenance.get("workflow", ""),
                task.provenance.get("initial_style", ""),
                task.provenance.get("information_mode", ""),
                task.provenance.get("goal_variant", ""),
            )
        )
        if structure != expected_structure:
            raise ValueError(f"{task.id} has inconsistent v0.3 structural provenance")
        if task.goal.terminal_behavior != "answer":
            raise ValueError(f"{task.id} must terminate with an answer")
        if "rl_compositional" not in task.categories:
            raise ValueError(f"{task.id} is missing the RL compositional category")
        action_horizon = len(task.goal.required_tools) + len(
            task.goal.required_clarifications
        )
        if not 2 <= action_horizon <= 8:
            raise ValueError(f"{task.id} action horizon is outside [2, 8]")
        if any(not 0.0 < fault.probability <= 1.0 for fault in task.fault_plan):
            raise ValueError(f"{task.id} has an invalid seeded fault probability")
        target = task.goal.final_state.get("target")
        tracking = task.goal.final_state.get("tracking")
        if task.goal.required_clarifications:
            kind_word = "人物" if tracking == "person" else "物体"
            if (
                task.goal.required_clarifications != ("target",)
                or f"指定{kind_word}" not in task.instruction
                or target in task.instruction
                or task.clarifications.get("target") != target
            ):
                raise ValueError(f"{task.id} has an invalid missing-target contract")
        elif "set_tracking" in task.goal.required_tools and tracking != "off":
            if not isinstance(target, str) or target not in task.instruction:
                raise ValueError(f"{task.id} hides an explicit tracking target")
        structures.add(structure)
        required_sequences.add(task.goal.required_tools)
    if len(tasks) >= 8 and (len(structures) < 2 or len(required_sequences) < 2):
        raise ValueError("v0.3 split lacks structural workflow diversity")


def _validate_v3_structural_holdouts(tasks: list[TaskSpec]) -> None:
    structures: dict[str, set[str]] = {split: set() for split in RL_SPLITS}
    for task in tasks:
        structures[task.split].add(task.provenance["structure"])
    for index, split in enumerate(RL_SPLITS):
        for other in RL_SPLITS[index + 1 :]:
            overlap = structures[split] & structures[other]
            if overlap:
                raise ValueError(
                    f"v0.3 structures overlap between {split} and {other}: {sorted(overlap)}"
                )


def _validate_v3_atomic_factor_coverage(tasks: list[TaskSpec]) -> None:
    factor_names = ("workflow", "initial_profile", "information_mode", "goal_shape")
    by_split = {
        split: {
            factor: {
                task.provenance[factor] for task in tasks if task.split == split
            }
            for factor in factor_names
        }
        for split in RL_SPLITS
    }
    for split in ("dev", "test"):
        for factor in factor_names:
            unseen = by_split[split][factor] - by_split["train"][factor]
            if unseen:
                raise ValueError(
                    f"v0.3 {split} introduces unseen atomic {factor} values: {sorted(unseen)}"
                )


def _v3_split_statistics(tasks: tuple[TaskSpec, ...]) -> dict[str, Any]:
    structures = sorted({task.provenance["structure"] for task in tasks})
    structure_payload = json.dumps(structures, ensure_ascii=False, separators=(",", ":"))
    return {
        "task_count": len(tasks),
        "structure_count": len(structures),
        "structure_ids_sha256": hashlib.sha256(structure_payload.encode("utf-8")).hexdigest(),
        "workflow_counts": dict(
            sorted(Counter(task.provenance["workflow"] for task in tasks).items())
        ),
        "initial_profile_counts": dict(
            sorted(Counter(task.provenance["initial_profile"] for task in tasks).items())
        ),
        "fault_plan_count": sum(bool(task.fault_plan) for task in tasks),
        "clarification_count": sum(bool(task.goal.required_clarifications) for task in tasks),
    }


def _validate_v3_manifest_metadata(
    manifest: dict[str, Any],
    tasks_by_split: dict[str, tuple[TaskSpec, ...]],
) -> None:
    root_seed = manifest.get("root_seed")
    if not isinstance(root_seed, int) or isinstance(root_seed, bool) or root_seed < 0:
        raise ValueError("v0.3 manifest root seed is invalid")
    if manifest.get("seed_scheme") != "split-base-plus-root-v0.1":
        raise ValueError("v0.3 manifest seed scheme is invalid")
    if manifest.get("structure_catalogue_version") != "v0.1":
        raise ValueError("v0.3 manifest structure catalogue version is invalid")
    for split, tasks in tasks_by_split.items():
        for index, task in enumerate(tasks):
            expected_seed = root_seed * 1_000_000 + {
                "train": 10_000,
                "dev": 20_000,
                "test": 30_000,
            }[split] + index + 1
            if task.seed != expected_seed:
                raise ValueError(f"{task.id} does not match the v0.3 manifest root seed")
    expected_statistics = {
        split: _v3_split_statistics(tasks_by_split[split]) for split in RL_SPLITS
    }
    if manifest.get("split_statistics") != expected_statistics:
        raise ValueError("v0.3 manifest split coverage statistics do not match task files")
