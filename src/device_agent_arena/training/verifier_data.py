from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from device_agent_arena.contracts import (
    ContractCatalog,
    ContractEngine,
    load_contract_catalog,
    rename_contract_catalog,
)
from device_agent_arena.domain.models import ToolCall, TrajectoryEvent
from device_agent_arena.harness.types import AgentContext


def build_verifier_records(
    contract_root: str | Path,
    *,
    train_records: int = 8000,
    dev_id_records: int = 1000,
    dev_xschema_records: int = 1000,
) -> list[dict[str, Any]]:
    for name, count in (
        ("train_records", train_records),
        ("dev_id_records", dev_id_records),
        ("dev_xschema_records", dev_xschema_records),
    ):
        if count < 5 or count % 5:
            raise ValueError(f"{name} must be a positive multiple of 5")
    root = Path(contract_root)
    room = load_contract_catalog(root / "meeting-room-v1.json")
    studio = load_contract_catalog(root / "studio-rig-v1.json")
    node = load_contract_catalog(root / "aliased-node-v1.json")
    train_surfaces = _training_aliases(room)
    studio_surfaces = _training_studio_aliases(studio)
    train_templates = (
        "Focus the {entity} room {value}.",
        "Please focus {value} for meeting {entity}.",
        "Before {entity} starts, focus {value}.",
        "Focus setting for {entity}: {value}.",
        "Can you focus the {entity} camera {value}?",
        "Meeting {entity}: focus {value}.",
        "Focus camera {value} in space {entity}.",
        "Set focus {value} for {entity}.",
    )
    dev_templates = (
        "For session {entity}, the requested focus is {value}.",
        "Focus must be {value} when {entity} begins.",
        "Could the {entity} setup focus {value} now?",
        "One change for {entity}: focus {value}.",
    )
    node_templates = (
        "Aim the {entity} node {value}.",
        "Please steer the {entity} node {value}.",
        "Before run {entity}, aim {value}.",
        "Steer {value} for node {entity}.",
    )
    studio_train_templates = (
        "Set the {entity} studio scene to {value}.",
        "Please use the {value} preset for studio {entity}.",
        "Before production {entity}, choose the {value} scene.",
        "Studio {entity}: scene preset {value}.",
    )
    studio_dev_templates = (
        "Production {entity} needs the {value} scene.",
        "Could the rig for {entity} use preset {value}?",
        "One studio change for {entity}: {value}.",
        "The requested scene on {entity} is {value}.",
    )
    room_train_records = (train_records // 10) * 5
    studio_train_records = train_records - room_train_records
    room_dev_records = (dev_id_records // 10) * 5
    studio_dev_records = dev_id_records - room_dev_records
    records: list[dict[str, Any]] = []
    records.extend(
        _build_split(
            split="train",
            surfaces=train_surfaces,
            templates=train_templates,
            entities=tuple(f"T{index:03d}" for index in range(96)),
            requested_records=room_train_records,
            operation_id="room.focus",
            values=("near", "far"),
            states=(None, {"room.paired": False}, {"room.paired": True}),
        )
    )
    for surface, surface_states, surface_records in zip(
        studio_surfaces,
        _studio_states(),
        _allocate_records(studio_train_records, len(studio_surfaces)),
        strict=True,
    ):
        records.extend(
            _build_split(
            split="train",
            surfaces=(surface,),
            templates=studio_train_templates,
            entities=tuple(f"S{index:03d}" for index in range(128)),
            requested_records=surface_records,
            operation_id="studio.scene",
            values=("portrait", "wide"),
            states=surface_states,
            )
        )
    records.extend(
        _build_split(
            split="dev_id",
            surfaces=train_surfaces,
            templates=dev_templates,
            entities=tuple(f"D{index:03d}" for index in range(64)),
            requested_records=room_dev_records,
            operation_id="room.focus",
            values=("near", "far"),
            states=(None, {"room.paired": False}, {"room.paired": True}),
        )
    )
    for surface, surface_states, surface_records in zip(
        studio_surfaces,
        _studio_states(),
        _allocate_records(studio_dev_records, len(studio_surfaces)),
        strict=True,
    ):
        records.extend(
            _build_split(
            split="dev_id",
            surfaces=(surface,),
            templates=studio_dev_templates,
            entities=tuple(f"V{index:03d}" for index in range(96)),
            requested_records=surface_records,
            operation_id="studio.scene",
            values=("portrait", "wide"),
            states=surface_states,
            )
        )
    records.extend(
        _build_split(
            split="dev_xschema",
            surfaces=(node,),
            templates=node_templates,
            entities=tuple(f"X{index:03d}" for index in range(96)),
            requested_records=dev_xschema_records,
            operation_id="node.aim",
            values=("north", "south"),
            states=(
                None,
                {"node.bound": False, "node.tuned": False},
                {"node.bound": True, "node.tuned": False},
                {"node.bound": True, "node.tuned": True},
            ),
        )
    )
    return records


def audit_verifier_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    negatives = [record for record in records if record["label"] == 0]
    counterfactual_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = record.get("counterfactual_key")
        if key is not None:
            counterfactual_groups[str(key)].append(record)
    flipping_keys = {
        key
        for key, group in counterfactual_groups.items()
        if {int(record["label"]) for record in group} == {0, 1}
    }
    flipping_records = sum(
        1 for record in records if record.get("counterfactual_key") in flipping_keys
    )
    train_families = sorted(
        {record["schema_family"] for record in records if record["split"] == "train"}
    )
    dev_xschema_families = sorted(
        {
            record["schema_family"]
            for record in records
            if record["split"] == "dev_xschema"
        }
    )
    hidden_names = {"goal", "required_tools", "fault_plan", "target_facts", "grader"}
    hidden_runtime_fields = sum(
        _count_forbidden(record["input"], hidden_names) for record in records
    )
    group_labels: dict[str, list[int]] = defaultdict(list)
    for record in records:
        group_labels[str(record["group_id"])].append(int(record["label"]))
    replay_consistent = sum(labels.count(1) == 1 for labels in group_labels.values())
    return {
        "record_count": len(records),
        "split_counts": dict(sorted(Counter(record["split"] for record in records).items())),
        "train_schema_families": train_families,
        "dev_xschema_families": dev_xschema_families,
        "schema_valid_negative_rate": (
            sum(bool(record["schema_valid"]) for record in negatives) / len(negatives)
        ),
        "invalid_syntax_negative_rate": (
            sum(not bool(record["schema_valid"]) for record in negatives) / len(negatives)
        ),
        "counterfactual_flip_record_rate": flipping_records / len(records),
        "duplicate_record_ids": len(records)
        - len({str(record["record_id"]) for record in records}),
        "duplicate_inputs": len(records) - len({str(record["text"]) for record in records}),
        "hidden_runtime_fields": hidden_runtime_fields,
        "label_replay_consistency": replay_consistent / len(group_labels),
        "negative_type_counts": dict(
            sorted(Counter(record["negative_type"] for record in negatives).items())
        ),
        "positive_position_counts": dict(
            sorted(
                Counter(
                    int(str(record["record_id"]).rsplit("-", 1)[1])
                    for record in records
                    if int(record["label"]) == 1
                ).items()
            )
        ),
    }


def export_verifier_records(
    records: list[dict[str, Any]],
    *,
    output_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_path)
    manifest = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    output.write_text(serialized, encoding="utf-8", newline="\n")
    audit = audit_verifier_records(records)
    receipt = {
        "dataset": output.as_posix(),
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "audit": audit,
        "runtime_input_fields": [
            "tools",
            "contracts",
            "instruction",
            "public_state",
            "successful_history",
            "latest_result",
            "candidate_action",
        ],
        "offline_label_only_fields": ["label", "oracle_fingerprint", "negative_type"],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def build_domain_verifier_records(
    *,
    split: str,
    catalog: ContractCatalog,
    templates: tuple[str, ...],
    entities: tuple[str, ...],
    requested_records: int,
    operation_id: str,
    values: tuple[str, str],
    states: tuple[dict[str, object] | None, ...],
) -> list[dict[str, Any]]:
    """Build label-replayable candidate groups for an externally supplied domain."""
    if requested_records < 5 or requested_records % 5:
        raise ValueError("requested_records must be a positive multiple of 5")
    return _build_split(
        split=split,
        surfaces=(catalog,),
        templates=templates,
        entities=entities,
        requested_records=requested_records,
        operation_id=operation_id,
        values=values,
        states=states,
    )


def _training_aliases(room: ContractCatalog) -> tuple[ContractCatalog, ...]:
    definitions = (
        (
            "room-alias-a",
            ("survey_space", "link_panel", "adjust_depth", "close_request"),
            ("depth", "note"),
        ),
        (
            "room-alias-b",
            ("probe_hall", "sync_console", "set_lens", "end_job"),
            ("range", "report"),
        ),
        (
            "room-alias-c",
            ("read_zone", "join_terminal", "tune_view", "complete_flow"),
            ("band", "message"),
        ),
    )
    operation_ids = ("room.inspect", "room.pair", "room.focus", "room.finish")
    return tuple(
        rename_contract_catalog(
            room,
            catalog_id=catalog_id,
            public_names=dict(zip(operation_ids, public_names, strict=True)),
            argument_names={
                "room.focus": {"level": argument_names[0]},
                "room.finish": {"summary": argument_names[1]},
            },
        )
        for catalog_id, public_names, argument_names in definitions
    )


def _training_studio_aliases(studio: ContractCatalog) -> tuple[ContractCatalog, ...]:
    definitions = (
        (
            "studio-alias-a",
            ("inspect_rig", "power_rig", "calibrate_rig", "set_scene", "finish_studio_task"),
            ("preset", "summary"),
            {
                "studio.powered": "studio.powered",
                "studio.calibrated": "studio.calibrated",
                "studio.scene": "studio.scene",
            },
        ),
        (
            "studio-alias-b",
            ("audit_stage", "energize_stage", "align_stage", "choose_layout", "close_stage"),
            ("layout", "report"),
            {
                "studio.powered": "media.active",
                "studio.calibrated": "media.aligned",
                "studio.scene": "media.layout",
            },
        ),
        (
            "studio-alias-c",
            ("check_booth", "wake_stack", "sync_stack", "apply_frame", "end_session"),
            ("frame", "message"),
            {
                "studio.powered": "booth.awake",
                "studio.calibrated": "booth.synced",
                "studio.scene": "booth.frame",
            },
        ),
    )
    operation_ids = (
        "studio.inspect",
        "studio.power",
        "studio.calibrate",
        "studio.scene",
        "studio.finish",
    )
    return tuple(
        rename_contract_catalog(
            studio,
            catalog_id=catalog_id,
            public_names=dict(zip(operation_ids, public_names, strict=True)),
            argument_names={
                "studio.scene": {"preset": argument_names[0]},
                "studio.finish": {"summary": argument_names[1]},
            },
            fact_names=fact_names,
        )
        for catalog_id, public_names, argument_names, fact_names in definitions
    )


def _studio_states() -> tuple[tuple[dict[str, object] | None, ...], ...]:
    return (
        (
            None,
            {"studio.powered": False, "studio.calibrated": False},
            {"studio.powered": True, "studio.calibrated": False},
            {"studio.powered": True, "studio.calibrated": True},
        ),
        (
            None,
            {"media.active": False, "media.aligned": False},
            {"media.active": True, "media.aligned": False},
            {"media.active": True, "media.aligned": True},
        ),
        (
            None,
            {"booth.awake": False, "booth.synced": False},
            {"booth.awake": True, "booth.synced": False},
            {"booth.awake": True, "booth.synced": True},
        ),
    )


def _allocate_records(total: int, buckets: int) -> tuple[int, ...]:
    groups, remainder = divmod(total // 5, buckets)
    return tuple(5 * (groups + int(index < remainder)) for index in range(buckets))


def _build_split(
    *,
    split: str,
    surfaces: tuple[ContractCatalog, ...],
    templates: tuple[str, ...],
    entities: tuple[str, ...],
    requested_records: int,
    operation_id: str,
    values: tuple[str, str],
    states: tuple[dict[str, object] | None, ...],
) -> list[dict[str, Any]]:
    requested_groups = requested_records // 5
    records: list[dict[str, Any]] = []
    group_index = 0
    for entity in entities:
        for template in templates:
            for value in values:
                instruction = template.format(entity=entity, value=value)
                for state_index, state in enumerate(states):
                    for surface in surfaces:
                        if group_index >= requested_groups:
                            return records
                        records.extend(
                            _build_candidate_group(
                                split=split,
                                catalog=surface,
                                group_index=group_index,
                                instruction=instruction,
                                operation_id=operation_id,
                                desired_value=value,
                                opposite_value=next(item for item in values if item != value),
                                public_state=state,
                                state_index=state_index,
                            )
                        )
                        group_index += 1
    raise ValueError(f"not enough unique {split} groups for {requested_records} records")


def _build_candidate_group(
    *,
    split: str,
    catalog: ContractCatalog,
    group_index: int,
    instruction: str,
    operation_id: str,
    desired_value: str,
    opposite_value: str,
    public_state: dict[str, object] | None,
    state_index: int,
) -> list[dict[str, Any]]:
    target = catalog.operation(operation_id)
    target_argument = str(target.tool["function"]["parameters"]["required"][0])
    target_call = ToolCall(target.public_name, {target_argument: desired_value})
    events = _observation_events(catalog, public_state)
    context = AgentContext(
        instruction=instruction,
        step=2 if events else 1,
        events=events,
        episode_id=f"offline-{split}-{catalog.catalog_id}-{group_index}",
    )
    correct = ContractEngine(catalog).start(
        instruction,
        episode_id=context.episode_id,
    ).intercept(context, target_call).action
    assert isinstance(correct, ToolCall)
    terminal = next(operation for operation in catalog.operations if operation.role == "terminal")
    terminal_argument = str(terminal.tool["function"]["parameters"]["required"][0])
    candidates = [
        correct,
        target_call,
        ToolCall(target.public_name, {target_argument: opposite_value}),
        ToolCall(terminal.public_name, {terminal_argument: "done"}),
        *(
            ToolCall(operation.public_name)
            for operation in catalog.operations
            if operation.role in {"inspect", "repair"}
        ),
    ]
    unique: list[ToolCall] = []
    seen: set[str] = set()
    for candidate in candidates:
        fingerprint = _call_fingerprint(candidate)
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(candidate)
    positive = next(candidate for candidate in unique if candidate == correct)
    negative_pool = [candidate for candidate in unique if candidate != correct]
    correct_role = catalog.operation_for_tool(correct.name).role
    wrong_repairs = [
        candidate
        for candidate in negative_pool
        if catalog.operation_for_tool(candidate.name).role == "repair"
    ]
    preferred = wrong_repairs if correct_role == "repair" else []
    negatives = [
        *preferred,
        *(candidate for candidate in negative_pool if candidate not in preferred),
    ][:4]
    while len(negatives) < 4:
        negatives.append(
            ToolCall(
                target.public_name,
                {target_argument: f"invalid-{len(negatives)}"},
            )
        )
    if group_index % 2 == 0:
        negatives[-1] = ToolCall(target.public_name, {})
    group_id = f"{split}-{catalog.catalog_id}-{group_index:05d}"
    visible_tools = catalog.model_tools()
    random.Random(
        int(hashlib.sha256(f"{group_id}:tools".encode()).hexdigest()[:16], 16)
    ).shuffle(visible_tools)
    public_contracts = catalog.public_contracts()
    random.Random(
        int(hashlib.sha256(f"{group_id}:contracts".encode()).hexdigest()[:16], 16)
    ).shuffle(public_contracts)
    ordered_candidates = [positive, *negatives]
    random.Random(
        int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:16], 16)
    ).shuffle(ordered_candidates)
    records: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(ordered_candidates):
        label = int(candidate == positive)
        schema_valid = _candidate_is_schema_valid(catalog, candidate)
        input_payload = {
            "tools": visible_tools,
            "contracts": public_contracts,
            "instruction": instruction,
            "public_state": dict(public_state or {}),
            "successful_history": [],
            "latest_result": None,
            "candidate_action": {
                "name": candidate.name,
                "arguments": candidate.arguments,
            },
        }
        candidate_key = _call_fingerprint(candidate)
        counterfactual_key = hashlib.sha256(
            json.dumps(
                [catalog.catalog_id, instruction, candidate_key],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "record_id": f"{group_id}-{candidate_index}",
                "split": split,
                "schema_family": catalog.catalog_id,
                "group_id": group_id,
                "state_variant": state_index,
                "input": input_payload,
                "text": _render_verifier_input(input_payload),
                "label": label,
                "schema_valid": schema_valid,
                "negative_type": (
                    None
                    if label
                    else _negative_type(catalog, candidate, target_call, schema_valid)
                ),
                "counterfactual_key": counterfactual_key,
                "oracle_fingerprint": _call_fingerprint(correct),
                "offline_replay_verified": True,
            }
        )
    return records


def _observation_events(
    catalog: ContractCatalog,
    public_state: dict[str, object] | None,
) -> tuple[TrajectoryEvent, ...]:
    if public_state is None:
        return ()
    inspect = next(operation for operation in catalog.operations if operation.role == "inspect")
    return (
        TrajectoryEvent(
            step=1,
            kind="tool_call",
            payload={"id": "observe-1", "name": inspect.public_name, "arguments": {}},
        ),
        TrajectoryEvent(
            step=1,
            kind="tool_result",
            payload={
                "id": "observe-1",
                "name": inspect.public_name,
                "ok": True,
                "code": "ok",
                "data": dict(public_state),
                "state_delta": {},
            },
        ),
    )


def _render_verifier_input(payload: dict[str, Any]) -> str:
    sections = (
        ("TOOLS", payload["tools"]),
        ("CONTRACTS", payload["contracts"]),
        ("INSTRUCTION", payload["instruction"]),
        ("PUBLIC_STATE", payload["public_state"]),
        ("SUCCESSFUL_HISTORY", payload["successful_history"]),
        ("LATEST_RESULT", payload["latest_result"]),
        ("CANDIDATE_ACTION", payload["candidate_action"]),
    )
    return "\n".join(
        f"[{name}]\n{json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
        for name, value in sections
    )


def _candidate_is_schema_valid(catalog: ContractCatalog, call: ToolCall) -> bool:
    try:
        operation = catalog.operation_for_tool(call.name)
    except KeyError:
        return False
    parameters = operation.tool["function"]["parameters"]
    properties = parameters.get("properties", {})
    if set(call.arguments) - set(properties):
        return False
    if set(parameters.get("required", ())) - set(call.arguments):
        return False
    return all(
        "enum" not in properties[name] or value in properties[name]["enum"]
        for name, value in call.arguments.items()
    )


def _negative_type(
    catalog: ContractCatalog,
    candidate: ToolCall,
    target: ToolCall,
    schema_valid: bool,
) -> str:
    if not schema_valid:
        return "invalid_arguments"
    operation = catalog.operation_for_tool(candidate.name)
    if operation.role == "terminal":
        return "premature_terminal"
    if candidate.name == target.name and candidate.arguments != target.arguments:
        return "wrong_argument"
    if candidate == target:
        return "precondition_violation"
    if operation.role == "inspect":
        return "redundant_or_mistimed_inspection"
    if operation.role == "repair":
        return "wrong_or_redundant_repair"
    return "schema_valid_distractor"


def _call_fingerprint(call: ToolCall) -> str:
    return json.dumps(
        [call.name, call.arguments],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _count_forbidden(value: object, forbidden: set[str]) -> int:
    if isinstance(value, dict):
        return sum(
            int(str(key) in forbidden) + _count_forbidden(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(_count_forbidden(item, forbidden) for item in value)
    return 0
