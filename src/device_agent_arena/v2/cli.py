from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from device_agent_arena.v1.io import load_tasks
from device_agent_arena.v2.composition_agents import (
    FinishImmediatelyCompositionAgent,
    OracleCompositionAgent,
    PublicReactiveCompositionAgent,
    PublicRecoveryCompositionAgent,
    RefuseImmediatelyCompositionAgent,
)
from device_agent_arena.v2.composition_comparison import (
    write_composition_paired_comparison,
)
from device_agent_arena.v2.composition_gate import (
    verify_composition_dev18_receipt,
    verify_composition_dev18_task_slice,
    write_composition_dev18_receipt,
)
from device_agent_arena.v2.composition_io import load_composition_tasks
from device_agent_arena.v2.composition_reporting import write_composition_report
from device_agent_arena.v2.composition_runner import (
    run_attested_composition_model_tasks,
    run_attested_layered_composition_model_tasks,
    run_composition_tasks,
)
from device_agent_arena.v2.composition_sft import (
    export_functiongemma_composition_recovery_sft,
    export_functiongemma_composition_sft,
    export_qwen_composition_curriculum_sft,
    export_qwen_failure_refinement_sft,
)
from device_agent_arena.v2.composition_suite import (
    validate_composition_suite,
    write_composition_suite,
)
from device_agent_arena.v2.gate import (
    verify_dev21_promotion_receipt,
    verify_dev21_task_slice,
    write_dev21_gate_receipt,
)
from device_agent_arena.v2.reporting import write_v2_report
from device_agent_arena.v2.schema_model_runner import run_attested_schema_model_tasks
from device_agent_arena.v2.sft import (
    audit_functiongemma_file,
    export_functiongemma_hard_state_sft,
    export_functiongemma_schema_sft,
    export_functiongemma_sft,
)
from device_agent_arena.v2.suite import (
    validate_schema_suite,
    write_schema_suite,
)


def _load_schema_suite_tasks(root: Path, split: str):
    validate_schema_suite(root)
    return load_tasks(root / f"{split}.jsonl")


def _load_composition_suite_tasks(root: Path, split: str):
    validate_composition_suite(root)
    return load_composition_tasks(root / f"{split}.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="device-agent-arena-v2")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "export-functiongemma-sft",
        help="convert frozen Train-only next-decision data to FunctionGemma native protocol",
    )
    export.add_argument("--oracle-data", required=True)
    export.add_argument("--grounded-data", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--manifest-output", required=True)
    schema_export = commands.add_parser(
        "export-functiongemma-schema-sft",
        help="build deterministic schema-randomized hard-case continuation data",
    )
    schema_export.add_argument("--data", required=True)
    schema_export.add_argument("--output", required=True)
    schema_export.add_argument("--manifest-output", required=True)
    hard_state_export = commands.add_parser(
        "export-functiongemma-hard-state-sft",
        help="build a targeted post-inspection and post-confirmation continuation curriculum",
    )
    hard_state_export.add_argument("--data", required=True)
    hard_state_export.add_argument("--output", required=True)
    hard_state_export.add_argument("--manifest-output", required=True)
    hard_state_export.add_argument("--replay-modulus", type=int, default=8)
    composition_export = commands.add_parser(
        "export-functiongemma-composition-sft",
        help="build Train-only native FunctionGemma data from composition trajectories",
    )
    composition_export.add_argument("--suite-root", required=True)
    composition_export.add_argument("--output", required=True)
    composition_export.add_argument("--manifest-output", required=True)
    composition_export.add_argument("--composition-repeats", type=int, default=2)
    composition_export.add_argument("--replay-data")
    composition_export.add_argument("--replay-modulus", type=int, default=8)
    composition_recovery_export = commands.add_parser(
        "export-functiongemma-composition-recovery-sft",
        help="build Train-only hard-state and deviation-recovery FunctionGemma data",
    )
    composition_recovery_export.add_argument("--suite-root", required=True)
    composition_recovery_export.add_argument("--output", required=True)
    composition_recovery_export.add_argument("--manifest-output", required=True)
    composition_recovery_export.add_argument(
        "--hard-state-repeats", type=int, default=4
    )
    composition_recovery_export.add_argument("--replay-data")
    composition_recovery_export.add_argument("--replay-modulus", type=int, default=8)
    qwen_composition_export = commands.add_parser(
        "export-qwen-composition-sft",
        help="build Train-only Qwen tool-native oracle and recovery curriculum",
    )
    qwen_composition_export.add_argument("--suite-root", required=True)
    qwen_composition_export.add_argument("--output", required=True)
    qwen_composition_export.add_argument("--manifest-output", required=True)
    qwen_composition_export.add_argument("--oracle-repeats", type=int, default=2)
    qwen_composition_export.add_argument("--recovery-repeats", type=int, default=4)
    qwen_refinement_export = commands.add_parser(
        "export-qwen-failure-refinement-sft",
        help="replay failed Train trajectories into Qwen hard-example refinement data",
    )
    qwen_refinement_export.add_argument("--suite-root", required=True)
    qwen_refinement_export.add_argument("--failure-run", required=True)
    qwen_refinement_export.add_argument("--replay-data", required=True)
    qwen_refinement_export.add_argument("--output", required=True)
    qwen_refinement_export.add_argument("--manifest-output", required=True)
    qwen_refinement_export.add_argument("--correction-repeats", type=int, default=8)
    qwen_refinement_export.add_argument("--hard-oracle-repeats", type=int, default=3)
    qwen_refinement_export.add_argument("--recovery-repeats", type=int, default=2)
    qwen_refinement_export.add_argument("--replay-modulus", type=int, default=4)
    audit = commands.add_parser(
        "audit-functiongemma-sft",
        help="render native records with the pinned processor and audit token lengths",
    )
    audit.add_argument("--data", required=True)
    audit.add_argument("--model-path", required=True)
    audit.add_argument("--max-length", type=int, required=True)
    audit.add_argument("--output", required=True)
    report = commands.add_parser(
        "report", help="aggregate a V2 run with per-schema condition metrics"
    )
    report.add_argument("--input", required=True)
    report.add_argument("--json-output", required=True)
    gate = commands.add_parser(
        "gate-dev21", help="evaluate the frozen V2.3 small-Dev promotion gate"
    )
    gate.add_argument("--config", required=True)
    gate.add_argument("--identity-report", required=True)
    gate.add_argument("--mixed-report", required=True)
    gate.add_argument("--identity-run", required=True)
    gate.add_argument("--mixed-run", required=True)
    gate.add_argument("--output", required=True)
    run_model = commands.add_parser(
        "run-model", help="run a local model with the V2 public protocol"
    )
    run_model.add_argument("--suite-root", required=True)
    run_model.add_argument("--split", choices=("train", "dev", "test"), required=True)
    run_model.add_argument("--output", required=True)
    run_model.add_argument("--model-id", required=True)
    run_model.add_argument("--model-revision")
    run_model.add_argument("--adapter-path")
    run_model.add_argument("--load-in-4bit", action="store_true")
    run_model.add_argument("--max-new-tokens", type=int, default=256)
    run_model.add_argument("--limit", type=int)
    run_model.add_argument(
        "--promotion-receipt",
        help="required PASS receipt when a Dev run contains more than 21 tasks",
    )
    run_model.add_argument("--shared-impact-guard", action="store_true")
    run_model.add_argument(
        "--schema-profile",
        choices=("identity", "schema-a", "schema-b", "mixed"),
        default="identity",
    )
    generate_suite = commands.add_parser(
        "generate-schema-suite",
        help="freeze Train/Dev for the V2 schema-shift slice while keeping Test seed absent",
    )
    generate_suite.add_argument("--output-root", required=True)
    generate_suite.add_argument("--root-seed", type=int, required=True)
    generate_suite.add_argument("--train-count", type=int, default=700)
    generate_suite.add_argument("--dev-count", type=int, default=210)
    generate_suite.add_argument("--test-count", type=int, default=420)
    validate = commands.add_parser("validate-schema-suite")
    validate.add_argument("--suite-root", required=True)
    generate_composition = commands.add_parser(
        "generate-composition-suite",
        help="freeze Train/Dev for dual-control, infeasibility, and cross-domain slices",
    )
    generate_composition.add_argument("--output-root", required=True)
    generate_composition.add_argument("--root-seed", type=int, required=True)
    generate_composition.add_argument("--train-count", type=int, default=90)
    generate_composition.add_argument("--dev-count", type=int, default=54)
    generate_composition.add_argument("--test-count", type=int, default=90)
    validate_composition = commands.add_parser("validate-composition-suite")
    validate_composition.add_argument("--suite-root", required=True)
    run_composition = commands.add_parser(
        "run-composition", help="run a deterministic composition baseline"
    )
    run_composition.add_argument("--suite-root", required=True)
    run_composition.add_argument("--split", choices=("train", "dev"), required=True)
    run_composition.add_argument(
        "--agent",
        choices=("oracle", "reactive", "recovery", "finish", "refuse"),
        required=True,
    )
    run_composition.add_argument("--output", required=True)
    run_composition.add_argument("--limit", type=int)
    report_composition = commands.add_parser(
        "report-composition", help="aggregate deterministic V2 composition metrics"
    )
    report_composition.add_argument("--input", required=True)
    report_composition.add_argument("--json-output", required=True)
    compare_composition = commands.add_parser(
        "compare-composition",
        help="run a same-task paired bootstrap comparison for composition conditions",
    )
    compare_composition.add_argument("--candidate-run", required=True)
    compare_composition.add_argument("--baseline-run", required=True)
    compare_composition.add_argument("--output", required=True)
    compare_composition.add_argument("--bootstrap-samples", type=int, default=10_000)
    compare_composition.add_argument("--seed", type=int, default=20260904)
    run_composition_model = commands.add_parser(
        "run-composition-model",
        help="run a local model on the frozen 18-task composition gate slice",
    )
    run_composition_model.add_argument("--suite-root", required=True)
    run_composition_model.add_argument(
        "--split", choices=("train", "dev"), required=True
    )
    run_composition_model.add_argument("--output", required=True)
    run_composition_model.add_argument("--model-id", required=True)
    run_composition_model.add_argument("--model-revision")
    run_composition_model.add_argument("--adapter-path")
    run_composition_model.add_argument("--load-in-4bit", action="store_true")
    run_composition_model.add_argument("--max-new-tokens", type=int, default=256)
    run_composition_model.add_argument("--limit", type=int)
    run_composition_model.add_argument("--start", type=int, default=0)
    run_composition_model.add_argument(
        "--promotion-receipt",
        help="required verified PASS receipt when composition Dev exceeds 18 tasks",
    )
    run_layered_model = commands.add_parser(
        "run-composition-layered-model",
        help="run the attested speculative FunctionGemma + Qwen composition condition",
    )
    run_layered_model.add_argument("--suite-root", required=True)
    run_layered_model.add_argument("--split", choices=("train", "dev"), required=True)
    run_layered_model.add_argument("--output", required=True)
    run_layered_model.add_argument("--router-model-id", required=True)
    run_layered_model.add_argument("--router-model-revision")
    run_layered_model.add_argument("--router-adapter-path")
    run_layered_model.add_argument("--router-load-in-4bit", action="store_true")
    run_layered_model.add_argument("--router-max-new-tokens", type=int, default=128)
    run_layered_model.add_argument("--planner-model-id", required=True)
    run_layered_model.add_argument("--planner-model-revision")
    run_layered_model.add_argument("--planner-adapter-path")
    run_layered_model.add_argument("--planner-load-in-4bit", action="store_true")
    run_layered_model.add_argument("--planner-max-new-tokens", type=int, default=256)
    run_layered_model.add_argument("--limit", type=int)
    run_layered_model.add_argument("--start", type=int, default=0)
    run_layered_model.add_argument(
        "--promotion-receipt",
        help="verified direct-planner PASS receipt required beyond leading Dev18",
    )
    composition_gate = commands.add_parser(
        "gate-composition-dev18",
        help="evaluate and authenticate the frozen composition Dev18 promotion gate",
    )
    composition_gate.add_argument("--config", required=True)
    composition_gate.add_argument("--report", required=True)
    composition_gate.add_argument("--run", required=True)
    composition_gate.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export-functiongemma-sft":
        result = export_functiongemma_sft(
            oracle_data=args.oracle_data,
            grounded_data=args.grounded_data,
            output=args.output,
            manifest_output=args.manifest_output,
        )
    elif args.command == "export-functiongemma-schema-sft":
        result = export_functiongemma_schema_sft(
            data=args.data,
            output=args.output,
            manifest_output=args.manifest_output,
        )
    elif args.command == "export-functiongemma-hard-state-sft":
        result = export_functiongemma_hard_state_sft(
            data=args.data,
            output=args.output,
            manifest_output=args.manifest_output,
            replay_modulus=args.replay_modulus,
        )
    elif args.command == "export-functiongemma-composition-sft":
        result = export_functiongemma_composition_sft(
            suite_root=args.suite_root,
            output=args.output,
            manifest_output=args.manifest_output,
            composition_repeats=args.composition_repeats,
            replay_data=args.replay_data,
            replay_modulus=args.replay_modulus,
        )
    elif args.command == "export-functiongemma-composition-recovery-sft":
        result = export_functiongemma_composition_recovery_sft(
            suite_root=args.suite_root,
            output=args.output,
            manifest_output=args.manifest_output,
            hard_state_repeats=args.hard_state_repeats,
            replay_data=args.replay_data,
            replay_modulus=args.replay_modulus,
        )
    elif args.command == "export-qwen-composition-sft":
        result = export_qwen_composition_curriculum_sft(
            suite_root=args.suite_root,
            output=args.output,
            manifest_output=args.manifest_output,
            oracle_repeats=args.oracle_repeats,
            recovery_repeats=args.recovery_repeats,
        )
    elif args.command == "export-qwen-failure-refinement-sft":
        result = export_qwen_failure_refinement_sft(
            suite_root=args.suite_root,
            failure_run=args.failure_run,
            replay_data=args.replay_data,
            output=args.output,
            manifest_output=args.manifest_output,
            correction_repeats=args.correction_repeats,
            hard_oracle_repeats=args.hard_oracle_repeats,
            recovery_repeats=args.recovery_repeats,
            replay_modulus=args.replay_modulus,
        )
    elif args.command == "audit-functiongemma-sft":
        result = audit_functiongemma_file(
            data=args.data,
            model_path=args.model_path,
            max_length=args.max_length,
            output=args.output,
        )
    elif args.command == "report":
        result = write_v2_report(args.input, args.json_output)
    elif args.command == "gate-dev21":
        result = write_dev21_gate_receipt(
            config_path=args.config,
            identity_report_path=args.identity_report,
            mixed_report_path=args.mixed_report,
            identity_run_path=args.identity_run,
            mixed_run_path=args.mixed_run,
            output_path=args.output,
        )
    elif args.command == "run-model":
        from device_agent_arena.agents.hf_backend import TransformersBackend

        root = Path(args.suite_root)
        tasks = _load_schema_suite_tasks(root, args.split)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            tasks = tasks[: args.limit]
        if args.split == "dev" and len(tasks) > 21:
            if args.promotion_receipt is None:
                raise ValueError(
                    "full Dev remains locked: provide a verified PASS --promotion-receipt"
                )
            backend_name = (
                args.model_id
                if args.adapter_path is None
                else f"{args.model_id}+{Path(args.adapter_path).name}"
            )
            receipt = verify_dev21_promotion_receipt(
                args.promotion_receipt,
                candidate_agent_id=f"v2-model:{backend_name}",
            )
            verify_dev21_task_slice(receipt, tasks)
        backend = TransformersBackend(
            model_id=args.model_id,
            model_revision=args.model_revision,
            adapter_path=args.adapter_path,
            load_in_4bit=args.load_in_4bit,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        path = run_attested_schema_model_tasks(
            tasks=tasks,
            backend=backend,
            split=args.split,
            output=args.output,
            schema_profile_id=args.schema_profile,
            shared_impact_guard=args.shared_impact_guard,
        )
        result = {"output": str(path), "task_count": len(tasks)}
    elif args.command == "generate-schema-suite":
        result = write_schema_suite(
            args.output_root,
            root_seed=args.root_seed,
            train_count=args.train_count,
            dev_count=args.dev_count,
            test_count=args.test_count,
        )
    elif args.command == "validate-schema-suite":
        result = validate_schema_suite(args.suite_root)
    elif args.command == "generate-composition-suite":
        result = write_composition_suite(
            args.output_root,
            root_seed=args.root_seed,
            train_count=args.train_count,
            dev_count=args.dev_count,
            test_count=args.test_count,
        )
    elif args.command == "validate-composition-suite":
        result = validate_composition_suite(args.suite_root)
    elif args.command == "run-composition":
        tasks = _load_composition_suite_tasks(Path(args.suite_root), args.split)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            tasks = tasks[: args.limit]
        factories = {
            "oracle": OracleCompositionAgent,
            "reactive": lambda _task: PublicReactiveCompositionAgent(),
            "recovery": lambda _task: PublicRecoveryCompositionAgent(),
            "finish": lambda _task: FinishImmediatelyCompositionAgent(),
            "refuse": lambda _task: RefuseImmediatelyCompositionAgent(),
        }
        agent_ids = {
            "oracle": "oracle-v2-composition-v0.1",
            "reactive": "public-reactive-v2-composition-v0.1",
            "recovery": "public-recovery-v2-composition-v0.1",
            "finish": "finish-immediately-v2-composition-v0.1",
            "refuse": "refuse-immediately-v2-composition-v0.1",
        }
        path = run_composition_tasks(
            tasks=tasks,
            agent_factory=factories[args.agent],
            agent_id=agent_ids[args.agent],
            split=args.split,
            output=args.output,
            run_metadata={"condition": "deterministic-baseline"},
        )
        result = {"output": str(path), "task_count": len(tasks)}
    elif args.command == "report-composition":
        result = write_composition_report(args.input, args.json_output)
    elif args.command == "compare-composition":
        result = write_composition_paired_comparison(
            candidate_path=args.candidate_run,
            baseline_path=args.baseline_run,
            output_path=args.output,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    elif args.command == "gate-composition-dev18":
        result = write_composition_dev18_receipt(
            config_path=args.config,
            report_path=args.report,
            run_path=args.run,
            output_path=args.output,
        )
    elif args.command == "run-composition-model":
        all_tasks = _load_composition_suite_tasks(Path(args.suite_root), args.split)
        if args.start < 0:
            raise ValueError("--start must be non-negative")
        limit = 18 if args.split == "dev" and args.limit is None else args.limit
        if limit is not None and limit < 1:
            raise ValueError("--limit must be positive")
        end = len(all_tasks) if limit is None else args.start + limit
        if args.split == "dev" and end > 18:
            if args.promotion_receipt is None:
                raise ValueError(
                    "composition Dev18 is the frozen candidate gate; broader Dev remains locked"
                )
            backend_name = (
                args.model_id
                if args.adapter_path is None
                else f"{args.model_id}+{Path(args.adapter_path).name}"
            )
            receipt = verify_composition_dev18_receipt(
                args.promotion_receipt,
                candidate_agent_id=f"v2-composition-model:{backend_name}",
            )
            verify_composition_dev18_task_slice(receipt, all_tasks)
        tasks = all_tasks[args.start:end]
        if not tasks:
            raise ValueError("composition task selection is empty")
        from device_agent_arena.agents.hf_backend import TransformersBackend

        backend = TransformersBackend(
            model_id=args.model_id,
            model_revision=args.model_revision,
            adapter_path=args.adapter_path,
            load_in_4bit=args.load_in_4bit,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        path = run_attested_composition_model_tasks(
            tasks=tasks,
            backend=backend,
            split=args.split,
            output=args.output,
        )
        result = {"output": str(path), "task_count": len(tasks)}
    elif args.command == "run-composition-layered-model":
        all_tasks = _load_composition_suite_tasks(Path(args.suite_root), args.split)
        if args.start < 0:
            raise ValueError("--start must be non-negative")
        limit = 18 if args.split == "dev" and args.limit is None else args.limit
        if limit is not None and limit < 1:
            raise ValueError("--limit must be positive")
        end = len(all_tasks) if limit is None else args.start + limit
        if args.split == "dev" and end > 18:
            if args.promotion_receipt is None:
                raise ValueError(
                    "layered comparison beyond Dev18 requires a planner promotion receipt"
                )
            planner_name = (
                args.planner_model_id
                if args.planner_adapter_path is None
                else (
                    f"{args.planner_model_id}+"
                    f"{Path(args.planner_adapter_path).name}"
                )
            )
            receipt = verify_composition_dev18_receipt(
                args.promotion_receipt,
                candidate_agent_id=f"v2-composition-model:{planner_name}",
            )
            verify_composition_dev18_task_slice(receipt, all_tasks)
        tasks = all_tasks[args.start:end]
        if not tasks:
            raise ValueError("composition task selection is empty")
        from device_agent_arena.agents.hf_backend import TransformersBackend

        router = TransformersBackend(
            model_id=args.router_model_id,
            model_revision=args.router_model_revision,
            adapter_path=args.router_adapter_path,
            load_in_4bit=args.router_load_in_4bit,
            max_new_tokens=args.router_max_new_tokens,
            temperature=0.0,
        )
        planner = TransformersBackend(
            model_id=args.planner_model_id,
            model_revision=args.planner_model_revision,
            adapter_path=args.planner_adapter_path,
            load_in_4bit=args.planner_load_in_4bit,
            max_new_tokens=args.planner_max_new_tokens,
            temperature=0.0,
        )
        path = run_attested_layered_composition_model_tasks(
            tasks=tasks,
            router_backend=router,
            planner_backend=planner,
            split=args.split,
            output=args.output,
        )
        result = {"output": str(path), "task_count": len(tasks)}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
