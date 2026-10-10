from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from device_agent_arena.agents.guarded import (
    VISIBLE_CONTRACT_GUARD_VERSION,
    VisibleContractGuardAgent,
)
from device_agent_arena.agents.hf_backend import TransformersBackend
from device_agent_arena.agents.model_agent import ModelAgent
from device_agent_arena.agents.oracle import OracleAgent
from device_agent_arena.agents.rule import RuleAgent
from device_agent_arena.artifacts.rl_evidence import (
    write_grpo_training_evidence,
    write_rl_evidence,
)
from device_agent_arena.artifacts.run_io import run_tasks
from device_agent_arena.contracts.benchmark import write_contract_dev_evidence
from device_agent_arena.contracts.freeze import (
    freeze_contractgraph_candidate,
    verify_contractgraph_freeze,
)
from device_agent_arena.contracts.rollout import (
    write_combined_rollout_promotion,
    write_rollout_comparison,
)
from device_agent_arena.contracts.sealed import write_sealed_suite
from device_agent_arena.dataset.generate import (
    generate_clarification_evaluation_tasks,
    generate_evaluation_tasks,
)
from device_agent_arena.dataset.rl_suite import (
    load_rl_task_selection,
    validate_rl_task_suite,
    write_rl_task_selection,
    write_rl_task_suite,
)
from device_agent_arena.domain.task_io import (
    CANONICAL_CATEGORIES,
    TaskValidationError,
    load_tasks,
    write_tasks,
)
from device_agent_arena.harness.runner import HarnessConfig
from device_agent_arena.prompts import BASE_SYSTEM_PROMPT
from device_agent_arena.reporting.aggregate import write_failure_slice, write_report
from device_agent_arena.reporting.compare import write_comparison
from device_agent_arena.training.data import (
    build_functiongemma_curriculum_v02,
    build_functiongemma_oec_replay_v03,
    convert_sft_records_for_functiongemma,
    export_sft_dataset,
    generate_rl_sft_records,
    generate_sft_records,
    load_verified_oec_records,
)
from device_agent_arena.training.dpo import (
    dry_run_dpo_training,
    load_dpo_training_config,
    train_dpo,
)
from device_agent_arena.training.grpo import (
    dry_run_grpo_training,
    load_grpo_training_config,
    train_grpo,
)
from device_agent_arena.training.oec import (
    build_oec_ablation_datasets,
    export_oec_sft,
    verify_oec_sft,
    write_oec_task_selection,
)
from device_agent_arena.training.preference import (
    export_fctr_preferences,
    verify_fctr_preferences,
)
from device_agent_arena.training.preference_eval import score_preferences
from device_agent_arena.training.qlora import dry_run_training, load_training_config, train_qlora
from device_agent_arena.training.sequential_dpo import train_sequential_dpo
from device_agent_arena.training.state_lattice import (
    audit_functiongemma_state_lattice_records,
    build_functiongemma_state_lattice_curriculum_v05,
)
from device_agent_arena.training.verifier import (
    dry_run_verifier_training,
    evaluate_verifier,
    load_verifier_config,
    train_verifier,
)
from device_agent_arena.training.verifier_data import (
    build_verifier_records,
    export_verifier_records,
)

DEFAULT_DATA_ROOT = Path("data") / "eval" / "v0.4"
DEFAULT_CLARIFICATION_DATA_ROOT = Path("data") / "eval" / "clarification-v0.1"
DEFAULT_SFT_PATH = Path("data") / "train" / "device_tool_sft.jsonl"
DEFAULT_SFT_MANIFEST = Path("data") / "train" / "manifest.json"
DEFAULT_FUNCTIONGEMMA_SFT_PATH = (
    Path("data") / "train" / "device_tool_sft_functiongemma_v0.1.jsonl"
)
DEFAULT_FUNCTIONGEMMA_SFT_MANIFEST = (
    Path("data") / "train" / "manifest_functiongemma_v0.1.json"
)
DEFAULT_FUNCTIONGEMMA_SOURCE_PATH = (
    Path("data") / "train" / "device_tool_rl_sft_v0.4.jsonl"
)
DEFAULT_FUNCTIONGEMMA_CURRICULUM_PATH = (
    Path("data") / "train" / "device_tool_sft_functiongemma_v0.2.jsonl"
)
DEFAULT_FUNCTIONGEMMA_CURRICULUM_MANIFEST = (
    Path("data") / "train" / "manifest_functiongemma_v0.2.json"
)
DEFAULT_RL_DATA_ROOT = Path("data") / "rl" / "v0.1"
DEFAULT_RL_SFT_PATH = Path("data") / "train" / "device_tool_rl_sft_v0.1.jsonl"
DEFAULT_RL_SFT_MANIFEST = Path("data") / "train" / "manifest_rl_sft_v0.1.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _deterministic_temperature(value: str) -> float:
    parsed = float(value)
    if parsed != 0.0:
        raise argparse.ArgumentTypeError("v0.2 supports deterministic temperature=0 only")
    return parsed


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate_data(output_root: Path, suite: str) -> int:
    tasks = (
        generate_clarification_evaluation_tasks()
        if suite == "clarification"
        else generate_evaluation_tasks()
    )
    for split in ("dev", "test"):
        write_tasks(output_root / f"{split}.jsonl", [task for task in tasks if task.split == split])
    print(
        json.dumps(
            {
                "generated": len(tasks),
                "output_root": str(output_root),
                "suite": suite,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _validate_data(data_root: Path, suite: str) -> int:
    try:
        dev = load_tasks(data_root / "dev.jsonl")
        test = load_tasks(data_root / "test.jsonl")
        tasks = (*dev, *test)
        if any(task.split != "dev" for task in dev):
            raise TaskValidationError("dev.jsonl contains non-dev tasks")
        if any(task.split != "test" for task in test):
            raise TaskValidationError("test.jsonl contains non-test tasks")
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise TaskValidationError("duplicate task id across splits")
        categories = {category for task in tasks for category in task.categories}
        expected_categories = (
            {"missing_parameter"} if suite == "clarification" else CANONICAL_CATEGORIES
        )
        if categories != expected_categories:
            raise TaskValidationError(
                "category coverage mismatch: expected "
                f"{sorted(expected_categories)}, got {sorted(categories)}"
            )
        expected_splits = (
            {"dev": 10, "test": 20} if suite == "clarification" else {"dev": 40, "test": 80}
        )
        actual_splits = Counter(task.split for task in tasks)
        if actual_splits != expected_splits:
            raise TaskValidationError(
                f"split size mismatch: expected {expected_splits}, got {dict(actual_splits)}"
            )
    except (OSError, TaskValidationError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 2

    summary = {
        "valid": True,
        "suite": suite,
        "task_count": len(tasks),
        "splits": dict(sorted(Counter(task.split for task in tasks).items())),
        "categories": dict(
            sorted(Counter(category for task in tasks for category in task.categories).items())
        ),
        "sha256": {
            "dev.jsonl": _digest(data_root / "dev.jsonl"),
            "test.jsonl": _digest(data_root / "test.jsonl"),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="device-agent-arena")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-data", help="generate frozen dev/test JSONL")
    generate.add_argument("--output-root", type=Path)
    generate.add_argument("--suite", choices=("full", "clarification"), default="full")

    validate = subparsers.add_parser("validate-data", help="validate frozen task data")
    validate.add_argument("--data-root", type=Path)
    validate.add_argument("--suite", choices=("full", "clarification"), default="full")

    generate_rl = subparsers.add_parser(
        "generate-rl-data", help="generate the frozen long-horizon RL task suite"
    )
    generate_rl.add_argument("--output-root", type=Path, default=DEFAULT_RL_DATA_ROOT)
    generate_rl.add_argument(
        "--version",
        choices=("v0.1", "v0.2", "v0.3"),
        default="v0.1",
    )
    generate_rl.add_argument("--root-seed", type=int, default=0)

    validate_rl = subparsers.add_parser(
        "validate-rl-data", help="validate the frozen long-horizon RL task suite"
    )
    validate_rl.add_argument("--data-root", type=Path, default=DEFAULT_RL_DATA_ROOT)

    freeze_selection = subparsers.add_parser(
        "freeze-rl-selection",
        help="freeze an ordered, source-bound RL task subset",
    )
    freeze_selection.add_argument("--data-root", type=Path, required=True)
    freeze_selection.add_argument("--split", choices=("train", "dev", "test"), required=True)
    freeze_selection.add_argument("--task-id", action="append", required=True)
    freeze_selection.add_argument("--policy", required=True)
    freeze_selection.add_argument("--output", type=Path, required=True)

    freeze_oec = subparsers.add_parser(
        "freeze-oec-selection",
        help="freeze the deterministic structure-balanced OEC train subset",
    )
    freeze_oec.add_argument("--data-root", type=Path, required=True)
    freeze_oec.add_argument("--count", type=_positive_int, default=32)
    freeze_oec.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="run a baseline over a frozen split")
    run.add_argument("--agent", choices=("rule", "oracle", "hf"), required=True)
    run.add_argument("--split", choices=("train", "dev", "test"), required=True)
    run.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-steps", type=_positive_int, default=8)
    run.add_argument("--timeout-seconds", type=_positive_float, default=120.0)
    run_subset = run.add_mutually_exclusive_group()
    run_subset.add_argument("--limit", type=_positive_int)
    run_subset.add_argument("--selection", type=Path)
    run.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    run.add_argument("--model-revision")
    run.add_argument("--adapter", type=Path)
    run.add_argument("--load-in-4bit", action="store_true")
    run.add_argument("--max-new-tokens", type=_positive_int, default=256)
    run.add_argument("--temperature", type=_deterministic_temperature, default=0.0)
    run.add_argument(
        "--visible-contract-guard",
        action="store_true",
        help="guard HF actions using only the visible instruction and observed tool results",
    )

    report = subparsers.add_parser("report", help="aggregate a raw run artifact")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--json-output", type=Path, required=True)
    report.add_argument("--markdown-output", type=Path, required=True)
    report.add_argument("--language", choices=("en", "zh-CN"), default="en")

    failures = subparsers.add_parser("failures", help="export failed trials as JSONL")
    failures.add_argument("--input", type=Path, required=True)
    failures.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare", help="compare two aggregate reports")
    compare.add_argument("--base-report", type=Path, required=True)
    compare.add_argument("--candidate-report", type=Path, required=True)
    compare.add_argument("--json-output", type=Path, required=True)
    compare.add_argument("--markdown-output", type=Path, required=True)
    compare.add_argument("--language", choices=("en", "zh-CN"), default="en")

    contract_dev = subparsers.add_parser(
        "run-contract-dev",
        help="run the synthetic cross-schema ContractGraph development regression",
    )
    contract_dev.add_argument("--contract-root", type=Path, default=Path("contracts"))
    contract_dev.add_argument("--output-root", type=Path, required=True)

    export_verifier = subparsers.add_parser(
        "export-contract-verifier-data",
        help="export train/dev-only hard-negative data for the ContractGraph verifier",
    )
    export_verifier.add_argument("--contract-root", type=Path, default=Path("contracts"))
    export_verifier.add_argument("--output", type=Path, required=True)
    export_verifier.add_argument("--manifest", type=Path, required=True)
    export_verifier.add_argument("--train-records", type=_positive_int, default=8000)
    export_verifier.add_argument("--dev-id-records", type=_positive_int, default=1000)
    export_verifier.add_argument("--dev-xschema-records", type=_positive_int, default=1000)

    train_verifier_parser = subparsers.add_parser(
        "train-contract-verifier",
        help="train the FunctionGemma state-action sequence-classification verifier",
    )
    train_verifier_parser.add_argument("--config", type=Path, required=True)
    train_verifier_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    train_verifier_parser.add_argument("--dry-run", action="store_true")

    evaluate_verifier_parser = subparsers.add_parser(
        "evaluate-contract-verifier",
        help="score frozen Dev-ID and Dev-XSchema candidate groups with a verifier adapter",
    )
    evaluate_verifier_parser.add_argument("--config", type=Path, required=True)
    evaluate_verifier_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    evaluate_verifier_parser.add_argument("--adapter", type=Path, required=True)
    evaluate_verifier_parser.add_argument("--output", type=Path, required=True)
    evaluate_verifier_parser.add_argument("--batch-size", type=_positive_int, default=16)
    evaluate_verifier_parser.add_argument("--evaluation-data", type=Path)
    evaluate_verifier_parser.add_argument("--split", action="append", dest="evaluation_splits")
    evaluate_verifier_parser.add_argument("--decision-threshold", type=float)

    freeze_contractgraph = subparsers.add_parser(
        "freeze-contractgraph-candidate",
        help="hash-bind the ContractGraph source, data, config, contracts, and adapter",
    )
    freeze_contractgraph.add_argument("--project-root", type=Path, default=Path.cwd())
    freeze_contractgraph.add_argument("--config", type=Path, required=True)
    freeze_contractgraph.add_argument("--adapter", type=Path, required=True)
    freeze_contractgraph.add_argument("--output", type=Path, required=True)
    freeze_contractgraph.add_argument("--evidence", type=Path, action="append", default=[])

    verify_contractgraph = subparsers.add_parser(
        "verify-contractgraph-freeze",
        help="verify that every hash-bound ContractGraph candidate file is unchanged",
    )
    verify_contractgraph.add_argument("--project-root", type=Path, default=Path.cwd())
    verify_contractgraph.add_argument("--receipt", type=Path, required=True)

    materialize_sealed = subparsers.add_parser(
        "materialize-contractgraph-sealed-suite",
        help="materialize a two-domain sealed suite from a post-freeze blueprint",
    )
    materialize_sealed.add_argument("--blueprint", type=Path, required=True)
    materialize_sealed.add_argument("--output-root", type=Path, required=True)

    rollout_comparison = subparsers.add_parser(
        "evaluate-contractgraph-rollouts",
        help="replay ranked candidate groups as stateful multi-step contract rollouts",
    )
    rollout_comparison.add_argument("--records", type=Path, required=True)
    rollout_comparison.add_argument("--score-report", type=Path, required=True)
    rollout_comparison.add_argument("--catalog", type=Path, required=True)
    rollout_comparison.add_argument("--split", required=True)
    rollout_comparison.add_argument("--schema-family")
    rollout_comparison.add_argument("--output", type=Path, required=True)

    combine_rollouts = subparsers.add_parser(
        "combine-contractgraph-rollouts",
        help="combine two or more sealed-domain rollout reports into one promotion gate",
    )
    combine_rollouts.add_argument("--input", type=Path, action="append", required=True)
    combine_rollouts.add_argument("--output", type=Path, required=True)

    export_sft = subparsers.add_parser("export-sft", help="export train-only Tool-Use SFT data")
    export_sft.add_argument("--output", type=Path, default=DEFAULT_SFT_PATH)
    export_sft.add_argument("--manifest", type=Path, default=DEFAULT_SFT_MANIFEST)
    export_sft.add_argument("--per-category", type=int, default=256)
    export_sft.add_argument("--missing-parameter-records", type=int)
    export_sft.add_argument(
        "--clarification-curriculum",
        choices=("v0.1", "v0.2"),
    )

    export_functiongemma = subparsers.add_parser(
        "export-functiongemma-sft",
        help="export train-only SFT data in FunctionGemma's native tool protocol",
    )
    export_functiongemma.add_argument("--output", type=Path, default=DEFAULT_FUNCTIONGEMMA_SFT_PATH)
    export_functiongemma.add_argument(
        "--manifest", type=Path, default=DEFAULT_FUNCTIONGEMMA_SFT_MANIFEST
    )

    export_functiongemma_curriculum = subparsers.add_parser(
        "export-functiongemma-curriculum",
        help="mix atomic and structural train-only supervision for FunctionGemma",
    )
    export_functiongemma_curriculum.add_argument(
        "--atomic-data", type=Path, default=DEFAULT_SFT_PATH
    )
    export_functiongemma_curriculum.add_argument(
        "--structural-data", type=Path, default=DEFAULT_FUNCTIONGEMMA_SOURCE_PATH
    )
    export_functiongemma_curriculum.add_argument(
        "--output", type=Path, default=DEFAULT_FUNCTIONGEMMA_CURRICULUM_PATH
    )
    export_functiongemma_curriculum.add_argument(
        "--manifest", type=Path, default=DEFAULT_FUNCTIONGEMMA_CURRICULUM_MANIFEST
    )

    export_functiongemma_oec = subparsers.add_parser(
        "export-functiongemma-oec-replay",
        help="mix unique structural replay with verified OEC repairs for FunctionGemma",
    )
    export_functiongemma_oec.add_argument("--replay-data", type=Path, required=True)
    export_functiongemma_oec.add_argument("--oec-data", type=Path, required=True)
    export_functiongemma_oec.add_argument("--oec-manifest", type=Path, required=True)
    export_functiongemma_oec.add_argument("--oec-audit", type=Path, required=True)
    export_functiongemma_oec.add_argument("--output", type=Path, required=True)
    export_functiongemma_oec.add_argument("--manifest", type=Path, required=True)
    export_functiongemma_state_lattice = subparsers.add_parser(
        "export-functiongemma-state-lattice",
        help="mix unique structural replay with balanced local state/action decisions",
    )
    export_functiongemma_state_lattice.add_argument(
        "--replay-data", type=Path, required=True
    )
    export_functiongemma_state_lattice.add_argument(
        "--per-action", type=_positive_int, default=32
    )
    export_functiongemma_state_lattice.add_argument("--output", type=Path, required=True)
    export_functiongemma_state_lattice.add_argument("--manifest", type=Path, required=True)
    export_functiongemma_state_lattice.add_argument("--audit", type=Path, required=True)
    export_functiongemma.add_argument(
        "--source-data",
        type=Path,
        default=DEFAULT_FUNCTIONGEMMA_SOURCE_PATH,
        help="10-tool train-only oracle curriculum to convert",
    )

    export_rl_sft = subparsers.add_parser(
        "export-rl-sft", help="export train-only long-horizon Oracle decisions"
    )
    export_rl_sft.add_argument("--data-root", type=Path, default=DEFAULT_RL_DATA_ROOT)
    export_rl_sft.add_argument("--output", type=Path, default=DEFAULT_RL_SFT_PATH)
    export_rl_sft.add_argument("--manifest", type=Path, default=DEFAULT_RL_SFT_MANIFEST)
    export_rl_sft.add_argument("--max-records", type=_positive_int, default=384)
    export_rl_sft.add_argument("--interaction-tools", action="store_true")
    export_rl_sft.add_argument(
        "--clarification-oversample",
        type=_positive_int,
        default=1,
    )

    export_oec = subparsers.add_parser(
        "export-oec-sft",
        help="export verified expert corrections over student train trajectories",
    )
    export_oec.add_argument("--tasks", type=Path, required=True)
    export_oec.add_argument("--selection", type=Path)
    export_oec.add_argument("--student-run", type=Path, required=True)
    export_oec.add_argument("--heldout-data-root", type=Path, required=True)
    export_oec.add_argument("--output", type=Path, required=True)
    export_oec.add_argument("--manifest", type=Path, required=True)
    export_oec.add_argument("--audit", type=Path, required=True)
    export_oec.add_argument(
        "--seam-policy",
        choices=("latest-v0.1", "multi-v0.2"),
        default="latest-v0.1",
    )

    build_oec_ablations = subparsers.add_parser(
        "build-oec-ablations",
        help="build equal-size static replay, FCTR, and mixed SFT datasets",
    )
    build_oec_ablations.add_argument("--corrections", type=Path, required=True)
    build_oec_ablations.add_argument("--correction-manifest", type=Path, required=True)
    build_oec_ablations.add_argument("--replay", type=Path, required=True)
    build_oec_ablations.add_argument("--replay-manifest", type=Path, required=True)
    build_oec_ablations.add_argument("--output-root", type=Path, required=True)
    build_oec_ablations.add_argument("--record-count", type=_positive_int, default=50)
    build_oec_ablations.add_argument("--seed", type=int, default=42)
    build_oec_ablations.add_argument("--prioritize-first-seams", action="store_true")

    verify_oec = subparsers.add_parser(
        "verify-oec-sft",
        help="replay OEC repairs and verify dataset isolation",
    )
    verify_oec.add_argument("--tasks", type=Path, required=True)
    verify_oec.add_argument("--selection", type=Path)
    verify_oec.add_argument("--student-run", type=Path, required=True)
    verify_oec.add_argument("--heldout-data-root", type=Path, required=True)
    verify_oec.add_argument("--dataset", type=Path, required=True)
    verify_oec.add_argument("--manifest", type=Path, required=True)
    verify_oec.add_argument("--audit", type=Path, required=True)
    verify_oec.add_argument("--output", type=Path, required=True)

    export_preferences = subparsers.add_parser(
        "export-fctr-preferences",
        help="export executable same-state Oracle/student preference pairs",
    )
    export_preferences.add_argument("--tasks", type=Path, required=True)
    export_preferences.add_argument("--selection", type=Path)
    export_preferences.add_argument("--student-run", type=Path, required=True)
    export_preferences.add_argument("--heldout-data-root", type=Path, required=True)
    export_preferences.add_argument("--output", type=Path, required=True)
    export_preferences.add_argument("--manifest", type=Path, required=True)
    export_preferences.add_argument("--audit", type=Path, required=True)
    export_preferences.add_argument(
        "--protocol", choices=("openai", "functiongemma"), default="openai"
    )

    verify_preferences = subparsers.add_parser(
        "verify-fctr-preferences",
        help="replay and verify executable FCTR preference pairs",
    )
    verify_preferences.add_argument("--tasks", type=Path, required=True)
    verify_preferences.add_argument("--selection", type=Path)
    verify_preferences.add_argument("--student-run", type=Path, required=True)
    verify_preferences.add_argument("--heldout-data-root", type=Path, required=True)
    verify_preferences.add_argument("--dataset", type=Path, required=True)
    verify_preferences.add_argument("--manifest", type=Path, required=True)
    verify_preferences.add_argument("--audit", type=Path, required=True)
    verify_preferences.add_argument("--output", type=Path, required=True)
    verify_preferences.add_argument(
        "--protocol", choices=("openai", "functiongemma"), default="openai"
    )

    score_preferences_parser = subparsers.add_parser(
        "score-fctr-preferences",
        help="score chosen/rejected preference pairs without updating model weights",
    )
    score_preferences_parser.add_argument("--dataset", type=Path, required=True)
    score_preferences_parser.add_argument("--manifest", type=Path, required=True)
    score_preferences_parser.add_argument("--model-id", required=True)
    score_preferences_parser.add_argument("--model-revision", required=True)
    score_preferences_parser.add_argument("--adapter", type=Path)
    score_preferences_parser.add_argument("--max-length", type=_positive_int, default=1280)
    score_preferences_parser.add_argument("--load-in-4bit", action="store_true")
    score_preferences_parser.add_argument(
        "--compute-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    score_preferences_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    score_preferences_parser.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train", help="validate or run the QLoRA recipe")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--project-root", type=Path, default=Path.cwd())
    train.add_argument("--dry-run", action="store_true")
    train.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts") / "training" / "qwen3-1.7b-dry-run.json",
    )

    train_preference = subparsers.add_parser(
        "train-dpo", help="validate or run state-conditioned preference QLoRA"
    )
    train_preference.add_argument("--config", type=Path, required=True)
    train_preference.add_argument("--project-root", type=Path, default=Path.cwd())
    train_preference.add_argument("--dry-run", action="store_true")
    train_preference.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts") / "training" / "fctr-dpo-dry-run.json",
    )

    train_sequential = subparsers.add_parser(
        "train-sequential-dpo",
        help="run exact-gradient memory-efficient sequential DPO",
    )
    train_sequential.add_argument("--config", type=Path, required=True)
    train_sequential.add_argument("--project-root", type=Path, default=Path.cwd())

    train_rl = subparsers.add_parser("train-rl", help="validate or run online trajectory GRPO")
    train_rl.add_argument("--config", type=Path, required=True)
    train_rl.add_argument("--project-root", type=Path, default=Path.cwd())
    train_rl.add_argument("--dry-run", action="store_true")
    train_rl.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts") / "training" / "grpo-dry-run.json",
    )

    verify_rl = subparsers.add_parser(
        "verify-rl-evidence",
        help="recompute an online-RL experiment and evaluate its promotion gate",
    )
    verify_rl.add_argument("--project-root", type=Path, default=Path.cwd())
    verify_rl.add_argument("--task-file", type=Path, required=True)
    verify_rl.add_argument("--task-selection", type=Path)
    verify_rl.add_argument("--training-config", type=Path, required=True)
    verify_rl.add_argument("--training-result", type=Path, required=True)
    verify_rl.add_argument("--base-run", type=Path, required=True)
    verify_rl.add_argument("--candidate-run", type=Path, required=True)
    verify_rl.add_argument("--base-report", type=Path, required=True)
    verify_rl.add_argument("--candidate-report", type=Path, required=True)
    verify_rl.add_argument("--comparison-report", type=Path, required=True)
    verify_rl.add_argument("--output", type=Path, required=True)
    verify_rl.add_argument("--minimum-nonzero-reward-std-rate", type=float, default=0.9)
    verify_rl.add_argument("--minimum-nonzero-gradient-rate", type=float, default=0.9)
    verify_training = subparsers.add_parser(
        "verify-grpo-training",
        help="verify a GRPO training smoke and its learning-signal gate",
    )
    verify_training.add_argument("--project-root", type=Path, default=Path.cwd())
    verify_training.add_argument("--training-config", type=Path, required=True)
    verify_training.add_argument("--training-result", type=Path, required=True)
    verify_training.add_argument("--output", type=Path, required=True)
    verify_training.add_argument(
        "--minimum-nonzero-reward-std-rate", type=float, default=0.9
    )
    verify_training.add_argument("--minimum-nonzero-gradient-rate", type=float, default=0.9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-contract-dev":
        comparison = write_contract_dev_evidence(
            contract_root=args.contract_root,
            output_root=args.output_root,
        )
        print(json.dumps(comparison, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "export-contract-verifier-data":
        records = build_verifier_records(
            args.contract_root,
            train_records=args.train_records,
            dev_id_records=args.dev_id_records,
            dev_xschema_records=args.dev_xschema_records,
        )
        receipt = export_verifier_records(
            records,
            output_path=args.output,
            manifest_path=args.manifest,
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "train-contract-verifier":
        config = load_verifier_config(args.config)
        result = (
            dry_run_verifier_training(config, project_root=args.project_root)
            if args.dry_run
            else train_verifier(config, project_root=args.project_root)
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "objective": result["objective"],
                    "hardware_gate": result.get("hardware_gate"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "evaluate-contract-verifier":
        result = evaluate_verifier(
            load_verifier_config(args.config),
            project_root=args.project_root,
            adapter_path=args.adapter,
            output_path=args.output,
            batch_size=args.batch_size,
            evaluation_data_path=args.evaluation_data,
            splits=(
                tuple(args.evaluation_splits)
                if args.evaluation_splits
                else ("dev_id", "dev_xschema")
            ),
            decision_threshold=args.decision_threshold,
        )
        print(json.dumps(result["by_split"], ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "freeze-contractgraph-candidate":
        receipt = freeze_contractgraph_candidate(
            project_root=args.project_root,
            config_path=args.config,
            adapter_path=args.adapter,
            output_path=args.output,
            evidence_paths=tuple(args.evidence),
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "verify-contractgraph-freeze":
        verification = verify_contractgraph_freeze(
            args.receipt,
            project_root=args.project_root,
        )
        print(json.dumps(verification, ensure_ascii=False, sort_keys=True))
        return 0 if verification["status"] == "valid" else 1
    if args.command == "materialize-contractgraph-sealed-suite":
        receipt = write_sealed_suite(args.blueprint, args.output_root)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "evaluate-contractgraph-rollouts":
        result = write_rollout_comparison(
            records_path=args.records,
            score_report_path=args.score_report,
            catalog_path=args.catalog,
            split=args.split,
            output_path=args.output,
            schema_family=args.schema_family,
        )
        print(
            json.dumps(
                {
                    "promotion_gate": result["promotion_gate"],
                    "paired_bootstrap": result["paired_bootstrap"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "combine-contractgraph-rollouts":
        result = write_combined_rollout_promotion(
            args.input,
            output_path=args.output,
        )
        print(json.dumps(result["promotion_gate"], ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "generate-rl-data":
        manifest = write_rl_task_suite(
            args.output_root,
            suite_version=args.version,
            root_seed=args.root_seed,
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "validate-rl-data":
        validation = validate_rl_task_suite(args.data_root)
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "freeze-rl-selection":
        manifest = write_rl_task_selection(
            source_task_path=args.data_root / f"{args.split}.jsonl",
            output_path=args.output,
            split=args.split,
            task_ids=args.task_id,
            selection_policy=args.policy,
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "freeze-oec-selection":
        manifest = write_oec_task_selection(
            source_task_path=args.data_root / "train.jsonl",
            output_path=args.output,
            count=args.count,
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "generate-data":
        default_root = (
            DEFAULT_CLARIFICATION_DATA_ROOT if args.suite == "clarification" else DEFAULT_DATA_ROOT
        )
        return _generate_data(args.output_root or default_root, args.suite)
    if args.command == "validate-data":
        default_root = (
            DEFAULT_CLARIFICATION_DATA_ROOT if args.suite == "clarification" else DEFAULT_DATA_ROOT
        )
        return _validate_data(args.data_root or default_root, args.suite)
    if args.command == "run":
        source_task_path = args.data_root / f"{args.split}.jsonl"
        tasks = load_tasks(source_task_path)
        if args.selection is not None:
            tasks = load_rl_task_selection(
                selection_path=args.selection,
                source_task_path=source_task_path,
                tasks=tasks,
                split=args.split,
            )
        if args.limit is not None:
            tasks = tasks[: args.limit]
        if args.agent == "rule":

            def agent_factory(_task):
                return RuleAgent()

            agent_id = "rule-v0.1"
            display_name = "Rule baseline"
            run_metadata = {
                "agent_type": "rule",
                "comparison_protocol": {"policy": agent_id},
            }
        elif args.agent == "oracle":
            agent_factory = OracleAgent
            agent_id = "oracle-v0.2"
            display_name = "Oracle upper bound"
            run_metadata = {
                "agent_type": "oracle",
                "comparison_protocol": {"policy": agent_id},
            }
        else:
            backend = TransformersBackend(
                model_id=args.model_id,
                model_revision=args.model_revision,
                adapter_path=args.adapter,
                load_in_4bit=args.load_in_4bit,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                generation_timeout_seconds=args.timeout_seconds,
            )

            def agent_factory(_task):
                model_agent = ModelAgent(backend)
                if args.visible_contract_guard:
                    return VisibleContractGuardAgent(model_agent)
                return model_agent

            adapter_label = f" + {args.adapter.name}" if args.adapter is not None else ""
            guard_label = (
                f"+visible-contract-guard-{VISIBLE_CONTRACT_GUARD_VERSION}"
                if args.visible_contract_guard
                else ""
            )
            agent_id = f"hf:{backend.name}{guard_label}"
            display_name = f"HF local · {args.model_id}{adapter_label}"
            if args.visible_contract_guard:
                display_name += " · visible-contract guard"
            backend_metadata = backend.provenance()
            prompt_sha256 = hashlib.sha256(BASE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
            protocol_keys = (
                "model_id",
                "model_revision",
                "requested_model_revision",
                "chat_template_sha256",
                "load_in_4bit",
                "max_new_tokens",
                "temperature",
                "generation_timeout_seconds",
                "gpu_name",
                "packages",
            )
            run_metadata = {
                "agent_type": "hf_guarded" if args.visible_contract_guard else "hf",
                "backend": backend_metadata,
                "prompt_sha256": prompt_sha256,
                "visible_contract_guard": args.visible_contract_guard,
                "comparison_protocol": {
                    **{key: backend_metadata[key] for key in protocol_keys},
                    "prompt_sha256": prompt_sha256,
                    "visible_contract_guard": (
                        VISIBLE_CONTRACT_GUARD_VERSION
                        if args.visible_contract_guard
                        else None
                    ),
                },
            }
        run_tasks(
            tasks=tasks,
            agent_factory=agent_factory,
            agent_id=agent_id,
            display_name=display_name,
            split=args.split,
            output=args.output,
            config=HarnessConfig(
                max_steps=args.max_steps,
                timeout_seconds=args.timeout_seconds,
            ),
            run_metadata=run_metadata,
        )
        print(json.dumps({"output": str(args.output), "tasks": len(tasks)}, ensure_ascii=False))
        return 0
    if args.command == "report":
        report = write_report(
            run_path=args.input,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
            language=args.language,
        )
        print(json.dumps({"task_success_rate": report["task_success_rate"]}, ensure_ascii=False))
        return 0
    if args.command == "failures":
        count = write_failure_slice(args.input, args.output)
        print(json.dumps({"failure_count": count}, ensure_ascii=False))
        return 0
    if args.command == "compare":
        comparison = write_comparison(
            base_path=args.base_report,
            candidate_path=args.candidate_report,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
            language=args.language,
        )
        print(
            json.dumps(
                {"task_success_delta_pp": comparison["task_success_delta_pp"]},
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "export-sft":
        category_counts = (
            {"missing_parameter": args.missing_parameter_records}
            if args.missing_parameter_records is not None
            else None
        )
        records = generate_sft_records(
            per_category=args.per_category,
            category_counts=category_counts,
            clarification_curriculum=args.clarification_curriculum,
        )
        manifest = export_sft_dataset(args.output, args.manifest, records)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "export-functiongemma-sft":
        source_records = [
            json.loads(line)
            for line in args.source_data.read_text(encoding="utf-8").splitlines()
            if line
        ]
        records = convert_sft_records_for_functiongemma(
            source_records
        )
        manifest = export_sft_dataset(args.output, args.manifest, records)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "export-functiongemma-curriculum":
        atomic_records = [
            json.loads(line)
            for line in args.atomic_data.read_text(encoding="utf-8").splitlines()
            if line
        ]
        structural_records = [
            json.loads(line)
            for line in args.structural_data.read_text(encoding="utf-8").splitlines()
            if line
        ]
        records = build_functiongemma_curriculum_v02(atomic_records, structural_records)
        manifest = export_sft_dataset(args.output, args.manifest, records)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "export-functiongemma-oec-replay":
        replay_records = [
            json.loads(line)
            for line in args.replay_data.read_text(encoding="utf-8").splitlines()
            if line
        ]
        oec_records = load_verified_oec_records(
            args.oec_data, args.oec_manifest, args.oec_audit
        )
        records = build_functiongemma_oec_replay_v03(replay_records, oec_records)
        manifest = export_sft_dataset(args.output, args.manifest, records)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "export-functiongemma-state-lattice":
        replay_records = [
            json.loads(line)
            for line in args.replay_data.read_text(encoding="utf-8").splitlines()
            if line
        ]
        records = build_functiongemma_state_lattice_curriculum_v05(
            replay_records,
            per_action=args.per_action,
        )
        manifest = export_sft_dataset(args.output, args.manifest, records)
        lattice_records = [
            record
            for record in records
            if record["metadata"].get("curriculum_axis") == "state_lattice"
        ]
        audit = audit_functiongemma_state_lattice_records(lattice_records)
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"manifest": manifest, "lattice_audit": audit},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "export-rl-sft":
        tasks = load_tasks(args.data_root / "train.jsonl")
        records = generate_rl_sft_records(
            tasks,
            max_records=args.max_records,
            interaction_tools=args.interaction_tools,
            clarification_oversample=args.clarification_oversample,
        )
        manifest = export_sft_dataset(args.output, args.manifest, records)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "export-oec-sft":
        manifest = export_oec_sft(
            task_path=args.tasks,
            task_selection_path=args.selection,
            student_run_path=args.student_run,
            heldout_data_root=args.heldout_data_root,
            output_path=args.output,
            manifest_path=args.manifest,
            audit_path=args.audit,
            seam_policy=args.seam_policy,
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-oec-ablations":
        receipt = build_oec_ablation_datasets(
            correction_path=args.corrections,
            correction_manifest_path=args.correction_manifest,
            replay_path=args.replay,
            replay_manifest_path=args.replay_manifest,
            output_root=args.output_root,
            record_count=args.record_count,
            seed=args.seed,
            prioritize_first_seams=args.prioritize_first_seams,
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "verify-oec-sft":
        evidence = verify_oec_sft(
            task_path=args.tasks,
            task_selection_path=args.selection,
            student_run_path=args.student_run,
            heldout_data_root=args.heldout_data_root,
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            audit_path=args.audit,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "export-fctr-preferences":
        manifest = export_fctr_preferences(
            task_path=args.tasks,
            task_selection_path=args.selection,
            student_run_path=args.student_run,
            heldout_data_root=args.heldout_data_root,
            output_path=args.output,
            manifest_path=args.manifest,
            audit_path=args.audit,
            protocol=args.protocol,
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "verify-fctr-preferences":
        evidence = verify_fctr_preferences(
            task_path=args.tasks,
            task_selection_path=args.selection,
            student_run_path=args.student_run,
            heldout_data_root=args.heldout_data_root,
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            audit_path=args.audit,
            protocol=args.protocol,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "score-fctr-preferences":
        result = score_preferences(
            project_root=args.project_root,
            dataset_path=str(args.dataset),
            manifest_path=str(args.manifest),
            model_id=args.model_id,
            model_revision=args.model_revision,
            adapter_path=str(args.adapter) if args.adapter is not None else None,
            max_length=args.max_length,
            load_in_4bit=args.load_in_4bit,
            compute_dtype=args.compute_dtype,
            output_path=args.output,
        )
        print(json.dumps(result["scores"], ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "train":
        config = load_training_config(args.config)
        if args.dry_run:
            result = dry_run_training(
                config, project_root=args.project_root, artifact_path=args.artifact
            )
        else:
            result = train_qlora(config, project_root=args.project_root)
        print(
            json.dumps(
                {"status": result["status"], "model_id": result["model_id"]},
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "train-dpo":
        config = load_dpo_training_config(args.config)
        result = (
            dry_run_dpo_training(
                config, project_root=args.project_root, artifact_path=args.artifact
            )
            if args.dry_run
            else train_dpo(config, project_root=args.project_root)
        )
        print(
            json.dumps(
                {"status": result["status"], "model_id": result["model_id"]},
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "train-sequential-dpo":
        config = load_dpo_training_config(args.config)
        result = train_sequential_dpo(config, project_root=args.project_root)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "model_id": result["model_id"],
                    "implementation": result["implementation"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "train-rl":
        config = load_grpo_training_config(args.config)
        if args.dry_run:
            result = dry_run_grpo_training(
                config, project_root=args.project_root, artifact_path=args.artifact
            )
        else:
            result = train_grpo(config, project_root=args.project_root)
        print(
            json.dumps(
                {"status": result["status"], "model_id": result["model_id"]},
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "verify-rl-evidence":
        evidence = write_rl_evidence(
            output_path=args.output,
            project_root=args.project_root,
            task_file_path=args.task_file,
            task_selection_path=args.task_selection,
            training_config_path=args.training_config,
            training_result_path=args.training_result,
            base_run_path=args.base_run,
            candidate_run_path=args.candidate_run,
            base_report_path=args.base_report,
            candidate_report_path=args.candidate_report,
            comparison_report_path=args.comparison_report,
            minimum_nonzero_reward_std_rate=args.minimum_nonzero_reward_std_rate,
            minimum_nonzero_gradient_rate=args.minimum_nonzero_gradient_rate,
        )
        print(
            json.dumps(
                {
                    "status": evidence["status"],
                    "promotion_gate": evidence["promotion_gate"]["status"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "verify-grpo-training":
        evidence = write_grpo_training_evidence(
            output_path=args.output,
            project_root=args.project_root,
            training_config_path=args.training_config,
            training_result_path=args.training_result,
            minimum_nonzero_reward_std_rate=args.minimum_nonzero_reward_std_rate,
            minimum_nonzero_gradient_rate=args.minimum_nonzero_gradient_rate,
        )
        print(
            json.dumps(
                {
                    "status": evidence["status"],
                    "learning_signal_gate": evidence["learning_signal_gate"]["status"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def entrypoint() -> None:
    raise SystemExit(main())
