from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from device_agent_arena.contracts.engine import ContractEngine
from device_agent_arena.contracts.schema import ContractCatalog, ToolContract, load_contract_catalog
from device_agent_arena.domain.models import ToolCall, TrajectoryEvent
from device_agent_arena.harness.types import AgentContext


@dataclass(frozen=True)
class ContractScenario:
    scenario_id: str
    split: str
    catalog_id: str
    instruction: str
    mode: str
    initial_facts: dict[str, object]
    proposed_call: ToolCall
    target_facts: dict[str, object]
    fault_operation_id: str | None = None


def build_schema_shift_dev(contract_root: str | Path) -> tuple[ContractScenario, ...]:
    """Build the public cross-schema development suite; this never emits Test records."""
    root = Path(contract_root)
    room = load_contract_catalog(root / "meeting-room-v1.json")
    node = load_contract_catalog(root / "aliased-node-v1.json")
    scenarios: list[ContractScenario] = []
    definitions = (
        (
            room,
            "room.focus",
            "level",
            "room.focus",
            ("near", "far"),
            (
                "Focus the room {value}.",
                "Please focus {value} for the meeting.",
                "Set the focus {value} before we start.",
                "Focus: {value}.",
            ),
            {"room.paired": True},
            {"room.paired": False},
        ),
        (
            node,
            "node.aim",
            "heading",
            "node.heading",
            ("north", "south"),
            (
                "Aim the node {value}.",
                "Please steer the node {value}.",
                "Aim {value} before the run.",
                "Steer: {value}.",
            ),
            {"node.bound": True, "node.tuned": True},
            {"node.bound": False, "node.tuned": False},
        ),
    )
    for (
        catalog,
        operation_id,
        argument_name,
        target_fact,
        values,
        templates,
        ready_facts,
        cold_facts,
    ) in definitions:
        operation = catalog.operation(operation_id)
        for value in values:
            opposite = next(candidate for candidate in values if candidate != value)
            for mode in ("ready", "cold", "transient", "wrong_argument"):
                for template_index, template in enumerate(templates):
                    proposed_value = opposite if mode == "wrong_argument" else value
                    scenarios.append(
                        ContractScenario(
                            scenario_id=(
                                f"dev-{catalog.catalog_id}-{value}-{mode}-{template_index:02d}"
                            ),
                            split="dev",
                            catalog_id=catalog.catalog_id,
                            instruction=template.format(value=value),
                            mode=mode,
                            initial_facts=dict(cold_facts if mode == "cold" else ready_facts),
                            proposed_call=ToolCall(
                                operation.public_name,
                                {argument_name: proposed_value},
                            ),
                            target_facts={target_fact: value},
                            fault_operation_id=(
                                operation.operation_id if mode == "transient" else None
                            ),
                        )
                    )
    return tuple(scenarios)


def evaluate_contract_scenarios(
    scenarios: tuple[ContractScenario, ...],
    *,
    contract_root: str | Path,
    guarded: bool,
) -> dict[str, Any]:
    root = Path(contract_root)
    catalogs = {
        path.stem: load_contract_catalog(path)
        for path in root.glob("*.json")
    }
    results = [
        _run_scenario(scenario, catalogs[scenario.catalog_id], guarded=guarded)
        for scenario in scenarios
    ]
    successes = sum(bool(result["success"]) for result in results)
    valid_calls = sum(int(result["valid_calls"]) for result in results)
    total_calls = sum(int(result["steps"]) for result in results)
    transient = [result for result in results if result["mode"] == "transient"]
    by_catalog = {
        catalog_id: {
            "task_count": len(group),
            "strict_success": sum(bool(item["success"]) for item in group),
        }
        for catalog_id in sorted({str(result["catalog_id"]) for result in results})
        if (
            group := [result for result in results if result["catalog_id"] == catalog_id]
        )
    }
    return {
        "suite": "schema-shift-dev-v0.3",
        "guarded": guarded,
        "task_count": len(results),
        "strict_success": successes,
        "strict_success_rate": successes / len(results),
        "argument_validity": valid_calls / total_calls,
        "contract_violations": sum(int(result["contract_violations"]) for result in results),
        "recovery_rate": (
            sum(bool(result["success"]) for result in transient) / len(transient)
            if transient
            else 0.0
        ),
        "mean_steps": mean(int(result["steps"]) for result in results),
        "hidden_runtime_fields": sum(
            int(result["hidden_runtime_fields"]) for result in results
        ),
        "mode_counts": dict(sorted(Counter(str(item["mode"]) for item in results).items())),
        "by_catalog": by_catalog,
        "results": results,
    }


def write_contract_dev_evidence(
    *,
    contract_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    contract_path = Path(contract_root)
    output_path = Path(output_root)
    scenarios = build_schema_shift_dev(contract_path)
    baseline = evaluate_contract_scenarios(
        scenarios,
        contract_root=contract_path,
        guarded=False,
    )
    guarded = evaluate_contract_scenarios(
        scenarios,
        contract_root=contract_path,
        guarded=True,
    )
    scenario_bytes = json.dumps(
        [asdict(scenario) for scenario in scenarios],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    comparison = {
        "suite": "schema-shift-dev-v0.3",
        "evidence_scope": "synthetic_dev_engineering_regression",
        "held_out_claim": False,
        "scripted_candidate_policy": True,
        "task_count": len(scenarios),
        "baseline_strict_success": baseline["strict_success"],
        "guarded_strict_success": guarded["strict_success"],
        "baseline_strict_success_rate": baseline["strict_success_rate"],
        "guarded_strict_success_rate": guarded["strict_success_rate"],
        "strict_success_delta_pp": round(
            100
            * (
                float(guarded["strict_success_rate"])
                - float(baseline["strict_success_rate"])
            ),
            2,
        ),
        "guarded_contract_violations": guarded["contract_violations"],
        "guarded_recovery_rate": guarded["recovery_rate"],
        "guarded_argument_validity": guarded["argument_validity"],
        "guarded_hidden_runtime_fields": guarded["hidden_runtime_fields"],
        "suite_sha256": hashlib.sha256(scenario_bytes).hexdigest(),
        "catalog_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(contract_path.glob("*.json"))
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("baseline.json", baseline),
        ("guarded.json", guarded),
        ("comparison.json", comparison),
    ):
        (output_path / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return comparison


def _run_scenario(
    scenario: ContractScenario,
    catalog: ContractCatalog,
    *,
    guarded: bool,
) -> dict[str, Any]:
    facts = dict(scenario.initial_facts)
    events: list[TrajectoryEvent] = []
    terminal = next(operation for operation in catalog.operations if operation.role == "terminal")
    proposals = deque(
        (
            scenario.proposed_call,
            ToolCall(terminal.public_name, {"summary": "done"}),
        )
    )
    session = (
        ContractEngine(catalog).start(scenario.instruction, episode_id=scenario.scenario_id)
        if guarded
        else None
    )
    fault_used = False
    terminated = False
    contract_violations = 0
    valid_calls = 0
    for step in range(1, 13):
        proposal = proposals.popleft() if proposals else ToolCall(
            terminal.public_name, {"summary": "done"}
        )
        context = AgentContext(
            instruction=scenario.instruction,
            step=step,
            events=tuple(events),
            episode_id=scenario.scenario_id,
        )
        action = session.intercept(context, proposal).action if session is not None else proposal
        assert isinstance(action, ToolCall)
        operation = catalog.operation_for_tool(action.name)
        valid_calls += int(_arguments_are_valid(operation, action.arguments))
        call_id = f"call-{step}"
        events.append(
            TrajectoryEvent(
                step=step,
                kind="tool_call",
                payload={"id": call_id, "name": action.name, "arguments": action.arguments},
            )
        )
        ok = True
        code = "ok"
        data: dict[str, object] = {}
        delta: dict[str, object] = {}
        if operation.role == "terminal":
            terminated = True
        elif scenario.fault_operation_id == operation.operation_id and not fault_used:
            fault_used = True
            ok = False
            code = "transient_failure"
        else:
            failed = next(
                (
                    precondition
                    for precondition in operation.preconditions
                    if facts.get(str(precondition["fact"])) != precondition.get("equals")
                ),
                None,
            )
            if failed is not None:
                ok = False
                code = "precondition_failed"
                contract_violations += 1
            elif operation.role == "inspect":
                data = dict(facts)
            else:
                delta = _apply_effects(operation, action.arguments, facts)
        events.append(
            TrajectoryEvent(
                step=step,
                kind="tool_result",
                payload={
                    "id": call_id,
                    "name": action.name,
                    "ok": ok,
                    "code": code,
                    "message": "",
                    "data": data,
                    "state_delta": delta,
                },
            )
        )
        if terminated:
            break
    hidden_names = {"goal", "required_tools", "fault_plan", "target_facts", "grader"}
    hidden_runtime_fields = sum(
        len(hidden_names.intersection(event.payload)) for event in events
    )
    success = terminated and all(
        facts.get(key) == value for key, value in scenario.target_facts.items()
    )
    return {
        "scenario_id": scenario.scenario_id,
        "catalog_id": scenario.catalog_id,
        "mode": scenario.mode,
        "success": success,
        "steps": len([event for event in events if event.kind == "tool_call"]),
        "valid_calls": valid_calls,
        "contract_violations": contract_violations,
        "hidden_runtime_fields": hidden_runtime_fields,
    }


def _apply_effects(
    operation: ToolContract,
    arguments: dict[str, object],
    facts: dict[str, object],
) -> dict[str, object]:
    delta: dict[str, object] = {}
    for effect in operation.effects:
        if "set" in effect:
            expression = effect["set"]
            delta[str(expression["fact"])] = expression["value"]
        else:
            expression = effect["set_from_argument"]
            delta[str(expression["fact"])] = arguments[str(expression["argument"])]
    facts.update(delta)
    return delta


def _arguments_are_valid(operation: ToolContract, arguments: dict[str, object]) -> bool:
    parameters = operation.tool["function"]["parameters"]
    properties = parameters.get("properties", {})
    if set(arguments) - set(properties):
        return False
    if set(parameters.get("required", ())) - set(arguments):
        return False
    for name, value in arguments.items():
        specification = properties[name]
        if specification.get("type") == "string" and not isinstance(value, str):
            return False
        if "enum" in specification and value not in specification["enum"]:
            return False
    return True
