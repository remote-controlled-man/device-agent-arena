from __future__ import annotations

import json
import random
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from device_agent_arena.contracts.schema import ContractCatalog, load_contract_catalog
from device_agent_arena.training.verifier import score_generic_candidate


def evaluate_ranked_rollouts(
    records: list[dict[str, Any]],
    scores: dict[str, float],
    *,
    catalog: ContractCatalog,
    policy: str,
    max_steps: int = 6,
    transient_every: int = 5,
) -> dict[str, Any]:
    supported_policies = {
        "learned",
        "no_ranker",
        "generic_rule",
        "oracle",
        "contract_masked_learned",
        "contract_masked_no_ranker",
        "contract_masked_generic_rule",
    }
    if policy not in supported_policies:
        raise ValueError(f"unsupported rollout policy: {policy}")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    target_by_instruction: dict[str, tuple[str, dict[str, object]]] = {}
    tasks: list[tuple[str, dict[str, object]]] = []
    for record in records:
        payload = record["input"]
        key = (str(payload["instruction"]), _state_key(payload["public_state"]))
        groups.setdefault(key, []).append(record)
        action = dict(payload["candidate_action"])
        try:
            action_role = catalog.operation_for_tool(str(action["name"])).role
        except KeyError:
            action_role = None
        if int(record["label"]) == 1 and action_role == "mutate":
            target_by_instruction[str(payload["instruction"])] = (
                str(action["name"]),
                dict(action["arguments"]),
            )
    for instruction, state_key in sorted(groups):
        tasks.append((instruction, json.loads(state_key)))

    observable_facts = {
        str(precondition["fact"])
        for operation in catalog.operations
        for precondition in operation.preconditions
    }
    outcomes: list[dict[str, Any]] = []
    for task_index, (instruction, initial_public) in enumerate(tasks):
        hidden = {fact: False for fact in observable_facts}
        hidden.update(initial_public)
        public = dict(initial_public)
        valid_calls = 0
        violations = 0
        trace: list[str] = []
        success = False
        fault_planned = transient_every > 0 and task_index % transient_every == 0
        fault_used = False
        for _ in range(max_steps):
            group = groups.get((instruction, _state_key(public)))
            if not group:
                break
            chosen = _choose(
                group,
                scores,
                policy,
                catalog=catalog,
                public_state=public,
                observable_facts=observable_facts,
            )
            if chosen is None:
                break
            action = dict(chosen["input"]["candidate_action"])
            trace.append(json.dumps(action, sort_keys=True))
            try:
                operation = catalog.operation_for_tool(str(action["name"]))
            except KeyError:
                break
            arguments = dict(action.get("arguments", {}))
            valid = _arguments_are_valid(operation.tool["function"]["parameters"], arguments)
            valid_calls += int(valid)
            if not valid:
                break
            if operation.role == "terminal":
                break
            failed = any(
                hidden.get(str(precondition["fact"])) != precondition.get("equals")
                for precondition in operation.preconditions
            )
            if failed:
                violations += 1
                continue
            if operation.role == "inspect":
                public = dict(hidden)
                continue
            if operation.role == "mutate" and fault_planned and not fault_used:
                fault_used = True
                continue
            for effect in operation.effects:
                expression = effect.get("set") or effect.get("set_from_argument")
                fact = str(expression["fact"])
                value = (
                    expression["value"]
                    if "set" in effect
                    else arguments[str(expression["argument"])]
                )
                hidden[fact] = value
                public[fact] = value
            target_name, target_arguments = target_by_instruction[instruction]
            if action["name"] == target_name and arguments == target_arguments:
                success = True
                break
        outcomes.append(
            {
                "instruction": instruction,
                "initial_public_state": initial_public,
                "success": success,
                "steps": len(trace),
                "valid_calls": valid_calls,
                "contract_violations": violations,
                "fault_planned": fault_planned,
                "fault_injected": fault_used,
                "trace": trace,
            }
        )
    total_calls = sum(int(item["steps"]) for item in outcomes)
    planned_fault_outcomes = [item for item in outcomes if item["fault_planned"]]
    injected_fault_outcomes = [item for item in outcomes if item["fault_injected"]]
    return {
        "policy": policy,
        "task_count": len(outcomes),
        "strict_success": sum(bool(item["success"]) for item in outcomes),
        "strict_success_rate": sum(bool(item["success"]) for item in outcomes) / len(outcomes),
        "argument_validity": (
            sum(int(item["valid_calls"]) for item in outcomes) / total_calls
            if total_calls
            else 0.0
        ),
        "contract_violation_rate": (
            sum(int(item["contract_violations"]) for item in outcomes) / total_calls
            if total_calls
            else 0.0
        ),
        "fault_planned_count": len(planned_fault_outcomes),
        "fault_injected_count": len(injected_fault_outcomes),
        "recovery_rate": (
            sum(bool(item["success"]) for item in planned_fault_outcomes)
            / len(planned_fault_outcomes)
            if planned_fault_outcomes
            else 0.0
        ),
        "conditional_recovery_rate": (
            sum(bool(item["success"]) for item in injected_fault_outcomes)
            / len(injected_fault_outcomes)
            if injected_fault_outcomes
            else 0.0
        ),
        "mean_steps": mean(int(item["steps"]) for item in outcomes),
        "outcomes": outcomes,
    }


def paired_bootstrap_delta(
    candidate: list[bool],
    baseline: list[bool],
    *,
    seed: int = 42,
    samples: int = 10_000,
) -> dict[str, float]:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired outcomes must have the same non-zero length")
    differences = [
        float(left) - float(right)
        for left, right in zip(candidate, baseline, strict=True)
    ]
    rng = random.Random(seed)
    estimates = sorted(
        mean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(samples)
    )
    return {
        "delta": mean(differences),
        "ci95_low": estimates[int(samples * 0.025)],
        "ci95_high": estimates[min(samples - 1, int(samples * 0.975))],
    }


def write_rollout_comparison(
    *,
    records_path: str | Path,
    score_report_path: str | Path,
    catalog_path: str | Path,
    split: str,
    output_path: str | Path,
    schema_family: str | None = None,
) -> dict[str, Any]:
    records_source = Path(records_path)
    score_report_source = Path(score_report_path)
    catalog_source = Path(catalog_path)
    records = [
        json.loads(line)
        for line in records_source.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [
        record
        for record in records
        if record["split"] == split
        and (schema_family is None or record["schema_family"] == schema_family)
    ]
    if not selected:
        raise ValueError(f"no rollout records for split: {split}")
    score_report = json.loads(score_report_source.read_text(encoding="utf-8"))
    scores = {
        str(item["record_id"]): float(item["score"])
        for item in score_report["scores"]
        if item["split"] == split
    }
    missing_scores = {str(record["record_id"]) for record in selected} - set(scores)
    if missing_scores:
        raise ValueError(f"score report is missing {len(missing_scores)} records")
    catalog = load_contract_catalog(catalog_source)
    policies = {
        name: evaluate_ranked_rollouts(selected, scores, catalog=catalog, policy=name)
        for name in ("no_ranker", "generic_rule", "learned", "oracle")
    }
    masked_policies = {
        name: evaluate_ranked_rollouts(selected, scores, catalog=catalog, policy=name)
        for name in (
            "contract_masked_no_ranker",
            "contract_masked_generic_rule",
            "contract_masked_learned",
        )
    }
    masked_baseline_name = max(
        ("contract_masked_no_ranker", "contract_masked_generic_rule"),
        key=lambda name: float(masked_policies[name]["strict_success_rate"]),
    )
    masked_bootstrap = paired_bootstrap_delta(
        [
            bool(item["success"])
            for item in masked_policies["contract_masked_learned"]["outcomes"]
        ],
        [
            bool(item["success"])
            for item in masked_policies[masked_baseline_name]["outcomes"]
        ],
    )
    baseline_name = max(
        ("no_ranker", "generic_rule"),
        key=lambda name: float(policies[name]["strict_success_rate"]),
    )
    candidate_outcomes = policies["learned"]["outcomes"]
    baseline_outcomes = policies[baseline_name]["outcomes"]
    bootstrap = paired_bootstrap_delta(
        [bool(item["success"]) for item in candidate_outcomes],
        [bool(item["success"]) for item in baseline_outcomes],
    )
    paired_successes = [
        (candidate, baseline)
        for candidate, baseline in zip(candidate_outcomes, baseline_outcomes, strict=True)
        if candidate["success"] and baseline["success"]
    ]
    paired_step_ratio = (
        mean(
            float(candidate["steps"]) / float(baseline["steps"])
            for candidate, baseline in paired_successes
        )
        if paired_successes
        else float("inf")
    )
    learned = policies["learned"]
    gate_checks = {
        "strict_success_delta_ge_0_10": bootstrap["delta"] >= 0.10,
        "paired_bootstrap_ci95_low_gt_0": bootstrap["ci95_low"] > 0,
        "argument_validity_eq_1": learned["argument_validity"] == 1.0,
        "contract_violation_rate_le_0_02": learned["contract_violation_rate"] <= 0.02,
        "recovery_rate_ge_0_70": learned["recovery_rate"] >= 0.70,
        "paired_success_step_ratio_le_1_25": paired_step_ratio <= 1.25,
    }
    result = {
        "status": "completed",
        "split": split,
        "schema_family": schema_family,
        "inputs": {
            "records": {
                "path": str(records_source),
                "sha256": sha256(records_source.read_bytes()).hexdigest(),
            },
            "score_report": {
                "path": str(score_report_source),
                "sha256": sha256(score_report_source.read_bytes()).hexdigest(),
            },
            "catalog": {
                "path": str(catalog_source),
                "sha256": sha256(catalog_source.read_bytes()).hexdigest(),
            },
        },
        "best_non_learned_baseline": baseline_name,
        "policies": policies,
        "post_hoc_contract_mask_diagnostic": {
            "promotion_eligible": False,
            "reason": (
                "Added after the frozen v0.8 raw-verifier gate failed; reported only "
                "to measure whether the learned ranker adds value inside the intended "
                "ContractGraph compound runtime."
            ),
            "best_non_learned_baseline": masked_baseline_name,
            "paired_bootstrap": masked_bootstrap,
            "policies": masked_policies,
        },
        "paired_bootstrap": bootstrap,
        "paired_success_step_ratio": paired_step_ratio,
        "promotion_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def write_combined_rollout_promotion(
    input_paths: list[str | Path],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    if len(input_paths) < 2:
        raise ValueError("combined sealed promotion requires at least two domains")
    domain_results = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in input_paths
    ]
    candidate: list[bool] = []
    baseline: list[bool] = []
    domains: dict[str, dict[str, Any]] = {}
    for result in domain_results:
        family = str(result["schema_family"])
        if family in domains:
            raise ValueError(f"duplicate rollout schema family: {family}")
        baseline_name = str(result["best_non_learned_baseline"])
        candidate.extend(
            bool(item["success"]) for item in result["policies"]["learned"]["outcomes"]
        )
        baseline.extend(
            bool(item["success"])
            for item in result["policies"][baseline_name]["outcomes"]
        )
        domains[family] = {
            "promotion_passed": bool(result["promotion_gate"]["passed"]),
            "learned_strict_success_rate": result["policies"]["learned"][
                "strict_success_rate"
            ],
            "baseline": baseline_name,
            "baseline_strict_success_rate": result["policies"][baseline_name][
                "strict_success_rate"
            ],
        }
    bootstrap = paired_bootstrap_delta(candidate, baseline)
    checks = {
        "all_domain_gates_passed": all(
            bool(item["promotion_passed"]) for item in domains.values()
        ),
        "combined_strict_success_delta_ge_0_10": bootstrap["delta"] >= 0.10,
        "combined_ci95_low_ge_0_03": bootstrap["ci95_low"] >= 0.03,
    }
    result = {
        "status": "completed",
        "domain_count": len(domains),
        "task_count": len(candidate),
        "domains": domains,
        "learned_strict_success_rate": sum(candidate) / len(candidate),
        "baseline_strict_success_rate": sum(baseline) / len(baseline),
        "paired_bootstrap": bootstrap,
        "promotion_gate": {"passed": all(checks.values()), "checks": checks},
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def _choose(
    group: list[dict[str, Any]],
    scores: dict[str, float],
    policy: str,
    *,
    catalog: ContractCatalog,
    public_state: dict[str, object],
    observable_facts: set[str],
) -> dict[str, Any] | None:
    if policy.startswith("contract_masked_"):
        group = [
            item
            for item in group
            if _is_contract_feasible(
                item,
                catalog=catalog,
                public_state=public_state,
                observable_facts=observable_facts,
            )
        ]
        policy = policy.removeprefix("contract_masked_")
    if not group:
        return None
    ordered = sorted(group, key=lambda item: str(item["record_id"]))
    if policy == "no_ranker":
        return ordered[0]
    if policy == "generic_rule":
        return max(
            ordered,
            key=lambda item: (
                score_generic_candidate(dict(item["input"])),
                str(item["record_id"]),
            ),
        )
    if policy == "oracle":
        return next(item for item in ordered if int(item["label"]) == 1)
    return max(ordered, key=lambda item: scores[str(item["record_id"])])


def _is_contract_feasible(
    record: dict[str, Any],
    *,
    catalog: ContractCatalog,
    public_state: dict[str, object],
    observable_facts: set[str],
) -> bool:
    action = dict(record["input"]["candidate_action"])
    try:
        operation = catalog.operation_for_tool(str(action["name"]))
    except KeyError:
        return False
    arguments = dict(action.get("arguments", {}))
    if not _arguments_are_valid(operation.tool["function"]["parameters"], arguments):
        return False
    if any(
        str(precondition["fact"]) not in public_state
        or public_state[str(precondition["fact"])] != precondition.get("equals")
        for precondition in operation.preconditions
    ):
        return False
    if operation.role == "inspect":
        return bool(observable_facts - set(public_state))
    if not operation.effects:
        return True
    for effect in operation.effects:
        expression = effect.get("set") or effect.get("set_from_argument")
        fact = str(expression["fact"])
        desired = (
            expression["value"]
            if "set" in effect
            else arguments[str(expression["argument"])]
        )
        if fact not in public_state or public_state[fact] != desired:
            return True
    return False


def _arguments_are_valid(schema: dict[str, Any], arguments: dict[str, object]) -> bool:
    properties = dict(schema.get("properties", {}))
    if set(arguments) - set(properties) or set(schema.get("required", ())) - set(arguments):
        return False
    return all(
        "enum" not in properties[name] or value in properties[name]["enum"]
        for name, value in arguments.items()
    )


def _state_key(state: dict[str, object]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))
