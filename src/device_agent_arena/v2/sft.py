from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from device_agent_arena.v2.protocol import functiongemma_record
from device_agent_arena.v2.schema_shift import schema_shift_functiongemma_record

_REPLACED_ORACLE_ARCHETYPES = {
    "shared_impact_accept",
    "shared_impact_decline",
    "dropped_write_recovery",
}
_SCHEMA_PROFILES = ("identity", "schema-a", "schema-b")
_HARD_CASE_ARCHETYPES = {"shared_impact_accept", "shared_impact_decline"}


def _token_ids(rendered: Any) -> list[int]:
    values = rendered.get("input_ids") if isinstance(rendered, Mapping) else rendered
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list) and values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("unexpected batched token IDs")
        values = values[0]
    if not isinstance(values, list) or not all(isinstance(item, int) for item in values):
        raise ValueError("FunctionGemma chat template did not return integer token IDs")
    return values


def audit_functiongemma_records(
    records: list[dict[str, Any]], *, processor: Any, max_length: int
) -> dict[str, Any]:
    if not records:
        raise ValueError("FunctionGemma token audit requires records")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    full_lengths: list[int] = []
    completion_lengths: list[int] = []
    prefix_mismatches = 0
    for record in records:
        prompt = _token_ids(
            processor.apply_chat_template(
                record["prompt"],
                tools=record["tools"],
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
            )
        )
        full = _token_ids(
            processor.apply_chat_template(
                [*record["prompt"], *record["completion"]],
                tools=record["tools"],
                tokenize=True,
                return_dict=True,
                add_generation_prompt=False,
            )
        )
        if full[: len(prompt)] != prompt:
            prefix_mismatches += 1
        full_lengths.append(len(full))
        completion_lengths.append(max(0, len(full) - len(prompt)))
    ordered = sorted(full_lengths)

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    truncation_count = sum(length > max_length for length in full_lengths)
    return {
        "format_version": "deviceops-ul-functiongemma-token-audit-v2.0",
        "record_count": len(records),
        "max_length": max_length,
        "min_tokens": ordered[0],
        "p50_tokens": percentile(0.50),
        "p95_tokens": percentile(0.95),
        "max_tokens": ordered[-1],
        "min_completion_tokens": min(completion_lengths),
        "max_completion_tokens": max(completion_lengths),
        "records_with_zero_completion_tokens": sum(
            length == 0 for length in completion_lengths
        ),
        "prompt_prefix_mismatch_count": prefix_mismatches,
        "completion_only_loss_verified": (
            prefix_mismatches == 0 and all(length > 0 for length in completion_lengths)
        ),
        "truncation_count": truncation_count,
        "truncation_rate": truncation_count / len(records),
    }


def audit_functiongemma_file(
    *,
    data: str | Path,
    model_path: str | Path,
    max_length: int,
    output: str | Path,
) -> dict[str, Any]:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional train dependency
        raise RuntimeError("transformers is required for FunctionGemma token audit") from exc
    source = Path(data)
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    result = audit_functiongemma_records(
        records,
        processor=processor,
        max_length=max_length,
    )
    result.update(
        {
            "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model_path": Path(model_path).as_posix(),
        }
    )
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite FunctionGemma token audit: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"SFT source is empty: {path}")
    for record in records:
        task_id = str(record.get("metadata", {}).get("task_id", ""))
        if "-test-" in task_id or "-dev-" in task_id:
            raise ValueError("FunctionGemma V2 export is Train-only")
    return records


def _archetype(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    explicit = metadata.get("archetype")
    if isinstance(explicit, str) and explicit:
        return explicit
    task_id = str(metadata.get("task_id", ""))
    return next((name for name in _REPLACED_ORACLE_ARCHETYPES if name in task_id), "")


def _replace_string_values(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_string_values(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_string_values(item, old, new) for key, item in value.items()}
    return value


def _remap_train_ids(record: dict[str, Any], prefix: str) -> dict[str, Any]:
    remapped = deepcopy(record)
    remapped["prompt"] = _replace_string_values(remapped["prompt"], "tr-", f"{prefix}-")
    remapped["completion"] = _replace_string_values(
        remapped["completion"], "tr-", f"{prefix}-"
    )
    remapped["metadata"]["entity_prefix"] = prefix
    return remapped


def _grounding_failures(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for record in records:
        prompt = json.dumps(record["prompt"], ensure_ascii=False, sort_keys=True)
        for call in record["completion"][0].get("tool_calls", []):
            arguments = call.get("function", {}).get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if not isinstance(arguments, dict):
                continue
            for field in ("device_id", "target_id", "operation_id"):
                value = arguments.get(field)
                if isinstance(value, str) and value not in prompt:
                    failures.append(
                        {
                            "task_id": record.get("metadata", {}).get("task_id"),
                            "field": field,
                            "value": value,
                        }
                    )
    return failures


def export_functiongemma_sft(
    *,
    oracle_data: str | Path,
    grounded_data: str | Path,
    output: str | Path,
    manifest_output: str | Path,
) -> dict[str, Any]:
    """Build a native multi-turn FunctionGemma curriculum from frozen Train-only data."""
    oracle_path = Path(oracle_data)
    grounded_path = Path(grounded_data)
    destination = Path(output)
    manifest_path = Path(manifest_output)
    for path in (destination, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite FunctionGemma V2 artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    raw_oracle = _read_records(oracle_path)
    oracle_records = [
        record
        for record in raw_oracle
        if _archetype(record) not in _REPLACED_ORACLE_ARCHETYPES
    ]
    sources = {"oracle": oracle_records, "grounded": _read_records(grounded_path)}
    records: list[dict[str, Any]] = []
    task_prefixes: dict[str, str] = {}
    for source_name, source_records in sources.items():
        for source_record in source_records:
            converted = functiongemma_record(source_record)
            converted["metadata"]["curriculum_source"] = source_name
            # Hugging Face Datasets infers one Arrow struct for the whole JSONL.
            # Keep optional curriculum metadata structurally identical across sources.
            converted["metadata"].setdefault("archetype", "")
            task_id = str(converted["metadata"].get("task_id", "unknown"))
            if task_id not in task_prefixes:
                task_prefixes[task_id] = f"g{len(task_prefixes):04d}"
            records.append(_remap_train_ids(converted, task_prefixes[task_id]))
    grounding_failures = _grounding_failures(records)
    if grounding_failures:
        raise ValueError(
            "FunctionGemma V2 curriculum contains ungrounded completion references: "
            f"{grounding_failures[:3]}"
        )
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    ).encode("utf-8")
    completion_names: Counter[str] = Counter()
    completion_without_call = 0
    for record in records:
        calls = record["completion"][0].get("tool_calls", [])
        if not calls:
            completion_without_call += 1
        else:
            completion_names[str(calls[0]["function"]["name"])] += 1
    manifest = {
        "format_version": "deviceops-ul-functiongemma-sft-manifest-v2.2.0",
        "generator": "deviceops-ul-functiongemma-grounded-remapped-v2.2.0",
        "record_count": len(records),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "source_record_counts": {
            name: len(source_records) for name, source_records in sorted(sources.items())
        },
        "raw_oracle_record_count": len(raw_oracle),
        "excluded_oracle_records": len(raw_oracle) - len(oracle_records),
        "replaced_oracle_archetypes": sorted(_REPLACED_ORACLE_ARCHETYPES),
        "grounding_failure_count": len(grounding_failures),
        "entity_prefix_count": len(task_prefixes),
        "model_visible_train_prefix_count": sum(
            "tr-" in json.dumps(
                [record["prompt"], record["completion"]], ensure_ascii=False
            )
            for record in records
        ),
        "source_sha256": {
            "oracle": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
            "grounded": hashlib.sha256(grounded_path.read_bytes()).hexdigest(),
        },
        "completion_tool_counts": dict(sorted(completion_names.items())),
        "completion_without_native_call_count": completion_without_call,
        "native_protocol": "functiongemma-native-v2",
        "train_only": True,
    }
    if completion_without_call:
        raise ValueError("FunctionGemma V2 curriculum contains a non-native completion")
    destination.write_bytes(serialized)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _primary_schema_profile(task_id: str) -> tuple[int, str]:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(_SCHEMA_PROFILES)
    return index, _SCHEMA_PROFILES[index]


def export_functiongemma_schema_sft(
    *,
    data: str | Path,
    output: str | Path,
    manifest_output: str | Path,
) -> dict[str, Any]:
    """Create a deterministic schema-randomized hard-case continuation curriculum."""
    source = Path(data)
    destination = Path(output)
    manifest_path = Path(manifest_output)
    for path in (destination, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite FunctionGemma V2 artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    source_records = _read_records(source)
    grounding_failures = _grounding_failures(source_records)
    if grounding_failures:
        raise ValueError(
            "schema curriculum source contains ungrounded completion references: "
            f"{grounding_failures[:3]}"
        )
    records: list[dict[str, Any]] = []
    profile_counts: Counter[str] = Counter()
    archetype_profile_counts: Counter[str] = Counter()
    hard_case_extra_count = 0
    for source_record in source_records:
        metadata = source_record.get("metadata", {})
        task_id = str(metadata.get("task_id", "unknown"))
        archetype = _archetype(source_record)
        primary_index, primary_profile = _primary_schema_profile(task_id)
        profiles = [primary_profile]
        if archetype in _HARD_CASE_ARCHETYPES:
            profiles.append(_SCHEMA_PROFILES[(primary_index + 1) % len(_SCHEMA_PROFILES)])
            hard_case_extra_count += 1
        for profile_id in profiles:
            shifted = schema_shift_functiongemma_record(source_record, profile_id)
            shifted["metadata"]["curriculum_stage"] = "schema-hardcase-v2.3"
            records.append(shifted)
            profile_counts[profile_id] += 1
            archetype_profile_counts[f"{archetype or 'unspecified'}::{profile_id}"] += 1

    completion_without_call = sum(
        not record["completion"][0].get("tool_calls") for record in records
    )
    if completion_without_call:
        raise ValueError("schema curriculum contains a non-native completion")
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    ).encode("utf-8")
    manifest = {
        "format_version": "deviceops-ul-functiongemma-sft-manifest-v2.3.0",
        "generator": "deviceops-ul-functiongemma-schema-hardcase-v2.3.0",
        "source_record_count": len(source_records),
        "record_count": len(records),
        "hard_case_extra_record_count": hard_case_extra_count,
        "hard_case_archetypes": sorted(_HARD_CASE_ARCHETYPES),
        "schema_profile_assignment": (
            "sha256(task_id) modulo 3; hard cases also use the next cyclic profile"
        ),
        "schema_profile_counts": dict(sorted(profile_counts.items())),
        "archetype_profile_counts": dict(sorted(archetype_profile_counts.items())),
        "source_grounding_failure_count": len(grounding_failures),
        "completion_without_native_call_count": completion_without_call,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "train_only": True,
    }
    destination.write_bytes(serialized)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _nested_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _nested_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _nested_values(child)]
    return [value]


def _hard_state_kind(record: dict[str, Any]) -> str | None:
    if _archetype(record) not in _HARD_CASE_ARCHETYPES:
        return None
    calls = record.get("completion", [{}])[0].get("tool_calls", [])
    if not calls:
        return None
    function = calls[0].get("function", {})
    name = str(function.get("name", ""))
    arguments = function.get("arguments", {})
    values = {str(value) for value in _nested_values(arguments)}
    if name == "request_confirmation":
        return "request_confirmation"
    if name == "refuse_task":
        return "decline_after_confirmation"
    if "pause_routine" in values:
        return "pause_confirmed_routine"
    prompt = json.dumps(record.get("prompt", []), ensure_ascii=False, sort_keys=True)
    if "set_mode" in values and "pause_routine" in prompt:
        return "set_mode_after_pause"
    return None


def _routine_device_pair(value: Any) -> tuple[str, str] | None:
    if isinstance(value, dict):
        routine_id = value.get("id")
        device_id = value.get("device_id")
        if isinstance(routine_id, str) and isinstance(device_id, str):
            return routine_id, device_id
        for child in value.values():
            pair = _routine_device_pair(child)
            if pair is not None:
                return pair
    elif isinstance(value, list):
        for child in value:
            pair = _routine_device_pair(child)
            if pair is not None:
                return pair
    return None


def _counterfactual_role_ids(
    record: dict[str, Any], *, variant: int
) -> dict[str, Any]:
    pair = _routine_device_pair(record.get("prompt", []))
    if pair is None:
        raise ValueError("critical shared-impact record lacks a grounded routine/device pair")
    routine_id, device_id = pair
    task_id = str(record.get("metadata", {}).get("task_id", "unknown"))
    tag = hashlib.sha256(f"{task_id}:{variant}".encode()).hexdigest()[:10]
    if variant == 0:
        new_device = f"cf-{tag}-routine-shaped"
        new_routine = f"cf-{tag}-device-shaped"
    else:
        new_device = f"cf-{tag}-alpha"
        new_routine = f"cf-{tag}-beta"
    remapped = _replace_string_values(record, device_id, new_device)
    remapped = _replace_string_values(remapped, routine_id, new_routine)
    remapped["metadata"] = dict(remapped.get("metadata", {}))
    remapped["metadata"]["counterfactual_role_ids"] = True
    remapped["metadata"]["counterfactual_variant"] = variant
    return remapped


def export_functiongemma_hard_state_sft(
    *,
    data: str | Path,
    output: str | Path,
    manifest_output: str | Path,
    replay_modulus: int = 8,
) -> dict[str, Any]:
    """Build a final targeted SFT stage around post-inspection/confirmation decisions."""
    if replay_modulus < 1:
        raise ValueError("replay_modulus must be positive")
    source = Path(data)
    destination = Path(output)
    manifest_path = Path(manifest_output)
    for path in (destination, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite FunctionGemma V2 artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    source_records = _read_records(source)
    records: list[dict[str, Any]] = []
    hard_state_counts: Counter[str] = Counter()
    critical_source_count = 0
    identity_extra_count = 0
    replay_count = 0
    for source_record in source_records:
        kind = _hard_state_kind(source_record)
        profile = str(source_record.get("metadata", {}).get("schema_profile", "unknown"))
        if kind is not None:
            critical_source_count += 1
            variants: list[dict[str, Any]] = [deepcopy(source_record)]
            if profile == "identity":
                variants.extend(
                    _counterfactual_role_ids(source_record, variant=variant)
                    for variant in range(2)
                )
                identity_extra_count += 2
            for repeat_index, item in enumerate(variants):
                item["metadata"] = dict(item.get("metadata", {}))
                item["metadata"].update(
                    {
                        "curriculum_stage": "hard-state-v2.4",
                        "hard_state_kind": kind,
                        "hard_state_repeat_index": repeat_index,
                        "replay": False,
                    }
                )
                item["metadata"].setdefault("counterfactual_role_ids", False)
                item["metadata"].setdefault("counterfactual_variant", -1)
                records.append(item)
                hard_state_counts[f"{profile}::{kind}"] += 1
            continue
        digest = hashlib.sha256(
            json.dumps(source_record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).digest()
        if int.from_bytes(digest[:8], "big") % replay_modulus == 0:
            item = deepcopy(source_record)
            item["metadata"] = dict(item.get("metadata", {}))
            item["metadata"].update(
                {
                    "curriculum_stage": "hard-state-v2.4",
                    "hard_state_kind": "replay",
                    "hard_state_repeat_index": 0,
                    "replay": True,
                    "counterfactual_role_ids": False,
                    "counterfactual_variant": -1,
                }
            )
            records.append(item)
            replay_count += 1
    if not records or not critical_source_count:
        raise ValueError("hard-state curriculum contains no critical records")
    completion_without_call = sum(
        not record["completion"][0].get("tool_calls") for record in records
    )
    if completion_without_call:
        raise ValueError("hard-state curriculum contains a non-native completion")
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    ).encode("utf-8")
    manifest = {
        "format_version": "deviceops-ul-functiongemma-sft-manifest-v2.4.1",
        "generator": "deviceops-ul-functiongemma-counterfactual-hard-state-v2.4.1",
        "source_record_count": len(source_records),
        "critical_source_record_count": critical_source_count,
        "identity_critical_extra_record_count": identity_extra_count,
        "replay_record_count": replay_count,
        "replay_modulus": replay_modulus,
        "record_count": len(records),
        "hard_state_counts": dict(sorted(hard_state_counts.items())),
        "completion_without_native_call_count": completion_without_call,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "train_only": True,
    }
    destination.write_bytes(serialized)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
