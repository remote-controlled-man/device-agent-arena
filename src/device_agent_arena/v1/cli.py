from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from device_agent_arena.v1.agents import (
    AlwaysAskAgent,
    DirectExecutionAgent,
    NaiveHeuristicAgent,
    NoToolAgent,
    OracleAgentV1,
)
from device_agent_arena.v1.failure_analysis import write_failure_analysis
from device_agent_arena.v1.io import load_tasks
from device_agent_arena.v1.model_agent import V1ModelAgent
from device_agent_arena.v1.reporting import (
    write_gate,
    write_paired_model_gate,
    write_paired_run_comparison,
    write_v1_report,
)
from device_agent_arena.v1.runner import run_v1_tasks
from device_agent_arena.v1.sft import (
    audit_sft_file,
    export_grounded_curriculum_sft,
    export_oracle_sft,
)
from device_agent_arena.v1.suite import materialize_test, validate_suite, write_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="device-agent-arena-v1")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="freeze Train/Dev and lock Test")
    generate.add_argument("--output-root", required=True)
    generate.add_argument("--root-seed", type=int, required=True)
    generate.add_argument("--train-count", type=int, default=98)
    generate.add_argument("--dev-count", type=int, default=28)
    generate.add_argument("--test-count", type=int, default=42)

    validate = commands.add_parser("validate", help="validate a frozen v1 suite")
    validate.add_argument("--suite-root", required=True)

    run = commands.add_parser("run", help="run a deterministic v1 baseline")
    run.add_argument("--suite-root", required=True)
    run.add_argument("--split", choices=("train", "dev", "test"), required=True)
    run.add_argument(
        "--agent",
        choices=("oracle", "heuristic", "direct", "always-ask", "no-tool"),
        required=True,
    )
    run.add_argument("--output", required=True)

    run_model = commands.add_parser("run-model", help="run a local HF model on V1")
    run_model.add_argument("--suite-root", required=True)
    run_model.add_argument("--split", choices=("train", "dev", "test"), required=True)
    run_model.add_argument("--output", required=True)
    run_model.add_argument("--model-id", required=True)
    run_model.add_argument("--model-revision")
    run_model.add_argument("--adapter-path")
    run_model.add_argument("--load-in-4bit", action="store_true")
    run_model.add_argument("--max-new-tokens", type=int, default=256)
    run_model.add_argument("--limit", type=int)

    report = commands.add_parser("report", help="aggregate a v1 run")
    report.add_argument("--input", required=True)
    report.add_argument("--json-output", required=True)

    analyze_failures = commands.add_parser(
        "analyze-failures", help="classify failures from an immutable raw V1 run"
    )
    analyze_failures.add_argument("--input", required=True)
    analyze_failures.add_argument("--output", required=True)

    export_sft = commands.add_parser(
        "export-sft", help="export Train-only oracle next-decision samples"
    )
    export_sft.add_argument("--suite-root", required=True)
    export_sft.add_argument("--output", required=True)
    export_sft.add_argument("--manifest-output", required=True)

    grounded_sft = commands.add_parser(
        "export-grounded-sft",
        help="export Train-only public-grounded correction curriculum",
    )
    grounded_sft.add_argument("--suite-root", required=True)
    grounded_sft.add_argument("--output", required=True)
    grounded_sft.add_argument("--manifest-output", required=True)

    audit_sft = commands.add_parser("audit-sft", help="measure rendered token lengths")
    audit_sft.add_argument("--data", required=True)
    audit_sft.add_argument("--model-path", required=True)
    audit_sft.add_argument("--max-length", type=int, required=True)
    audit_sft.add_argument("--output", required=True)

    gate = commands.add_parser("gate", help="evaluate the preregistered Dev gate")
    gate.add_argument("--suite-root", required=True)
    gate.add_argument("--oracle-report", required=True)
    gate.add_argument("--heuristic-report", required=True)
    gate.add_argument("--direct-report", required=True)
    gate.add_argument("--output", required=True)

    compare = commands.add_parser("compare", help="paired Dev model-vs-baseline gate")
    compare.add_argument("--candidate-run", required=True)
    compare.add_argument("--baseline-run", required=True)
    compare.add_argument("--bootstrap-samples", type=int, default=10_000)
    compare.add_argument("--seed", type=int, default=20260830)
    compare.add_argument("--output", required=True)

    compare_runs = commands.add_parser(
        "compare-runs", help="paired comparison on any identical frozen split"
    )
    compare_runs.add_argument("--candidate-run", required=True)
    compare_runs.add_argument("--baseline-run", required=True)
    compare_runs.add_argument("--bootstrap-samples", type=int, default=10_000)
    compare_runs.add_argument("--seed", type=int, default=20260830)
    compare_runs.add_argument("--output", required=True)

    unlock = commands.add_parser(
        "materialize-test", help="materialize Test after a passing paired Dev gate"
    )
    unlock.add_argument("--suite-root", required=True)
    unlock.add_argument("--model-gate", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        result = write_suite(
            args.output_root,
            root_seed=args.root_seed,
            train_count=args.train_count,
            dev_count=args.dev_count,
            test_count=args.test_count,
        )
    elif args.command == "validate":
        result = validate_suite(args.suite_root)
    elif args.command == "run":
        root = Path(args.suite_root)
        validate_suite(root)
        tasks = load_tasks(root / f"{args.split}.jsonl")
        factories = {
            "oracle": OracleAgentV1,
            "heuristic": lambda _task: NaiveHeuristicAgent(),
            "direct": lambda _task: DirectExecutionAgent(),
            "always-ask": lambda _task: AlwaysAskAgent(),
            "no-tool": lambda _task: NoToolAgent(),
        }
        ids = {
            "oracle": "oracle-v1.0",
            "heuristic": "naive-heuristic-v1.0",
            "direct": "direct-execution-v1.0",
            "always-ask": "always-ask-v1.0",
            "no-tool": "no-tool-v1.0",
        }
        path = run_v1_tasks(
            tasks=tasks,
            agent_factory=factories[args.agent],
            agent_id=ids[args.agent],
            split=args.split,
            output=args.output,
        )
        result = {"output": str(path), "task_count": len(tasks)}
    elif args.command == "report":
        result = write_v1_report(args.input, args.json_output)
    elif args.command == "analyze-failures":
        result = write_failure_analysis(args.input, args.output)
    elif args.command == "run-model":
        from device_agent_arena.agents.hf_backend import TransformersBackend

        root = Path(args.suite_root)
        validate_suite(root)
        tasks = load_tasks(root / f"{args.split}.jsonl")
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            tasks = tasks[: args.limit]
        backend = TransformersBackend(
            model_id=args.model_id,
            model_revision=args.model_revision,
            adapter_path=args.adapter_path,
            load_in_4bit=args.load_in_4bit,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        path = run_v1_tasks(
            tasks=tasks,
            agent_factory=lambda _task: V1ModelAgent(backend),
            agent_id=f"v1-model:{backend.name}",
            split=args.split,
            output=args.output,
            run_metadata=backend.provenance(),
            progress_callback=lambda completed, total: print(
                json.dumps(
                    {
                        "event": "v1_progress",
                        "completed": completed,
                        "total": total,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            ),
        )
        result = {"output": str(path), "task_count": len(tasks)}
    elif args.command == "export-sft":
        root = Path(args.suite_root)
        validate_suite(root)
        result = export_oracle_sft(
            load_tasks(root / "train.jsonl"),
            output=args.output,
            manifest_output=args.manifest_output,
        )
    elif args.command == "audit-sft":
        result = audit_sft_file(
            data_path=args.data,
            model_path=args.model_path,
            max_length=args.max_length,
            output_path=args.output,
        )
    elif args.command == "export-grounded-sft":
        root = Path(args.suite_root)
        validate_suite(root)
        result = export_grounded_curriculum_sft(
            load_tasks(root / "train.jsonl"),
            output=args.output,
            manifest_output=args.manifest_output,
        )
    elif args.command == "gate":
        result = write_gate(
            oracle_path=args.oracle_report,
            heuristic_path=args.heuristic_report,
            direct_path=args.direct_report,
            suite_root=args.suite_root,
            output_path=args.output,
        )
    elif args.command == "compare":
        result = write_paired_model_gate(
            candidate_path=args.candidate_run,
            baseline_path=args.baseline_run,
            output_path=args.output,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    elif args.command == "compare-runs":
        result = write_paired_run_comparison(
            candidate_path=args.candidate_run,
            baseline_path=args.baseline_run,
            output_path=args.output,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    elif args.command == "materialize-test":
        result = materialize_test(
            args.suite_root,
            model_gate_path=args.model_gate,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
