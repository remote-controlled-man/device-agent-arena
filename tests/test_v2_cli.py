from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from device_agent_arena.v2 import cli


def test_run_model_cannot_start_dev210_with_fail_receipt(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = [SimpleNamespace(id=f"task-{index}") for index in range(210)]
    monkeypatch.setattr(cli, "_load_schema_suite_tasks", lambda _root, _split: tasks)
    receipt = tmp_path / "fail.json"
    receipt.write_text(json.dumps({"status": "fail"}), encoding="utf-8")

    with pytest.raises(ValueError, match="did not pass"):
        cli.main(
            [
                "run-model",
                "--suite-root",
                str(tmp_path),
                "--split",
                "dev",
                "--output",
                str(tmp_path / "run.jsonl"),
                "--model-id",
                "candidate",
                "--promotion-receipt",
                str(receipt),
            ]
        )


def test_composition_cli_generates_runs_and_reports_oracle(tmp_path) -> None:
    suite = tmp_path / "composition-suite"
    run = tmp_path / "oracle.jsonl"
    report = tmp_path / "oracle-report.json"

    assert (
        cli.main(
            [
                "generate-composition-suite",
                "--output-root",
                str(suite),
                "--root-seed",
                "20260902",
                "--train-count",
                "18",
                "--dev-count",
                "18",
                "--test-count",
                "36",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "run-composition",
                "--suite-root",
                str(suite),
                "--split",
                "dev",
                "--agent",
                "oracle",
                "--output",
                str(run),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "report-composition",
                "--input",
                str(run),
                "--json-output",
                str(report),
            ]
        )
        == 0
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["safe_strict_success_rate"] == 1.0
    assert payload["task_count"] == 18


def test_composition_model_cannot_consume_more_than_frozen_dev18(tmp_path) -> None:
    suite = tmp_path / "composition-suite"
    cli.main(
        [
            "generate-composition-suite",
            "--output-root",
            str(suite),
            "--root-seed",
            "20260902",
            "--train-count",
            "18",
            "--dev-count",
            "54",
            "--test-count",
            "90",
        ]
    )

    with pytest.raises(ValueError, match="Dev18"):
        cli.main(
            [
                "run-composition-model",
                "--suite-root",
                str(suite),
                "--split",
                "dev",
                "--output",
                str(tmp_path / "forbidden.jsonl"),
                "--model-id",
                "candidate",
                "--limit",
                "19",
            ]
        )


def test_composition_cli_exports_train_only_functiongemma_sft(tmp_path) -> None:
    suite = tmp_path / "composition-suite"
    output = tmp_path / "composition-sft.jsonl"
    manifest = tmp_path / "composition-sft.manifest.json"
    cli.main(
        [
            "generate-composition-suite",
            "--output-root",
            str(suite),
            "--root-seed",
            "20260902",
            "--train-count",
            "18",
            "--dev-count",
            "18",
            "--test-count",
            "36",
        ]
    )

    assert (
        cli.main(
            [
                "export-functiongemma-composition-sft",
                "--suite-root",
                str(suite),
                "--output",
                str(output),
                "--manifest-output",
                str(manifest),
                "--composition-repeats",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["record_count"] > 0
    assert payload["dev_data_used"] is False
    assert payload["test_data_used"] is False


def test_composition_cli_exports_recovery_curriculum(tmp_path) -> None:
    suite = tmp_path / "composition-suite"
    output = tmp_path / "composition-recovery.jsonl"
    manifest = tmp_path / "composition-recovery.manifest.json"
    cli.main(
        [
            "generate-composition-suite",
            "--output-root",
            str(suite),
            "--root-seed",
            "20260902",
            "--train-count",
            "18",
            "--dev-count",
            "18",
            "--test-count",
            "36",
        ]
    )

    assert (
        cli.main(
            [
                "export-functiongemma-composition-recovery-sft",
                "--suite-root",
                str(suite),
                "--output",
                str(output),
                "--manifest-output",
                str(manifest),
                "--hard-state-repeats",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["recovery_state_source_count"] > 0
    assert payload["dev_data_used"] is False
    assert payload["test_data_used"] is False


def test_composition_cli_exports_qwen_curriculum(tmp_path) -> None:
    suite = tmp_path / "composition-suite"
    output = tmp_path / "qwen-composition.jsonl"
    manifest = tmp_path / "qwen-composition.manifest.json"
    cli.main(
        [
            "generate-composition-suite",
            "--output-root",
            str(suite),
            "--root-seed",
            "20260902",
            "--train-count",
            "18",
            "--dev-count",
            "18",
            "--test-count",
            "36",
        ]
    )

    assert (
        cli.main(
            [
                "export-qwen-composition-sft",
                "--suite-root",
                str(suite),
                "--output",
                str(output),
                "--manifest-output",
                str(manifest),
                "--oracle-repeats",
                "1",
                "--recovery-repeats",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["protocol"] == "qwen-tool-native-v2-composition"
    assert payload["dev_data_used"] is False
    assert payload["test_data_used"] is False


def test_composition_cli_dispatches_qwen_failure_refinement(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_export(**kwargs):
        captured.update(kwargs)
        return {"record_count": 7}

    monkeypatch.setattr(cli, "export_qwen_failure_refinement_sft", fake_export)
    assert (
        cli.main(
            [
                "export-qwen-failure-refinement-sft",
                "--suite-root",
                str(tmp_path / "suite"),
                "--failure-run",
                str(tmp_path / "run.jsonl"),
                "--replay-data",
                str(tmp_path / "base.jsonl"),
                "--output",
                str(tmp_path / "refinement.jsonl"),
                "--manifest-output",
                str(tmp_path / "refinement.manifest.json"),
                "--correction-repeats",
                "9",
            ]
        )
        == 0
    )
    assert captured["correction_repeats"] == 9
    assert captured["hard_oracle_repeats"] == 3
    assert captured["recovery_repeats"] == 2
    assert captured["replay_modulus"] == 4


def test_composition_cli_dispatches_layered_model_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import device_agent_arena.agents.hf_backend as hf_backend

    captured = {}

    class FakeBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_run(**kwargs):
        captured.update(kwargs)
        return tmp_path / "layered.jsonl"

    tasks = [SimpleNamespace(id="train-1"), SimpleNamespace(id="train-2")]
    monkeypatch.setattr(cli, "_load_composition_suite_tasks", lambda _root, _split: tasks)
    monkeypatch.setattr(hf_backend, "TransformersBackend", FakeBackend)
    monkeypatch.setattr(cli, "run_attested_layered_composition_model_tasks", fake_run)

    assert (
        cli.main(
            [
                "run-composition-layered-model",
                "--suite-root",
                str(tmp_path),
                "--split",
                "train",
                "--output",
                str(tmp_path / "layered.jsonl"),
                "--router-model-id",
                "router",
                "--planner-model-id",
                "planner",
                "--router-max-new-tokens",
                "128",
                "--planner-max-new-tokens",
                "256",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert len(captured["tasks"]) == 1
    assert captured["router_backend"].kwargs["model_id"] == "router"
    assert captured["router_backend"].kwargs["max_new_tokens"] == 128
    assert captured["planner_backend"].kwargs["model_id"] == "planner"
    assert captured["planner_backend"].kwargs["max_new_tokens"] == 256


def test_composition_cli_dispatches_paired_comparison(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_compare(**kwargs):
        captured.update(kwargs)
        return {"task_count": 18}

    monkeypatch.setattr(cli, "write_composition_paired_comparison", fake_compare)
    assert (
        cli.main(
            [
                "compare-composition",
                "--candidate-run",
                str(tmp_path / "layered.jsonl"),
                "--baseline-run",
                str(tmp_path / "direct.jsonl"),
                "--output",
                str(tmp_path / "comparison.json"),
                "--bootstrap-samples",
                "1000",
                "--seed",
                "7",
            ]
        )
        == 0
    )
    assert captured["bootstrap_samples"] == 1000
    assert captured["seed"] == 7


def test_composition_cli_uses_receipt_for_unseen_dev_tail(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import device_agent_arena.agents.hf_backend as hf_backend

    captured = {}
    tasks = [SimpleNamespace(id=f"dev-{index}") for index in range(54)]

    class FakeBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_verify(_path, *, candidate_agent_id):
        captured["candidate_agent_id"] = candidate_agent_id
        return {"task_count": 18}

    def fake_slice(_receipt, received_tasks):
        captured["slice_task_count"] = len(received_tasks)

    def fake_run(**kwargs):
        captured["selected"] = [task.id for task in kwargs["tasks"]]
        return tmp_path / "tail.jsonl"

    monkeypatch.setattr(cli, "_load_composition_suite_tasks", lambda _root, _split: tasks)
    monkeypatch.setattr(cli, "verify_composition_dev18_receipt", fake_verify)
    monkeypatch.setattr(cli, "verify_composition_dev18_task_slice", fake_slice)
    monkeypatch.setattr(hf_backend, "TransformersBackend", FakeBackend)
    monkeypatch.setattr(cli, "run_attested_composition_model_tasks", fake_run)

    assert (
        cli.main(
            [
                "run-composition-model",
                "--suite-root",
                str(tmp_path),
                "--split",
                "dev",
                "--output",
                str(tmp_path / "tail.jsonl"),
                "--model-id",
                "qwen",
                "--adapter-path",
                str(tmp_path / "adapter"),
                "--start",
                "18",
                "--limit",
                "36",
                "--promotion-receipt",
                str(tmp_path / "receipt.json"),
            ]
        )
        == 0
    )
    assert captured["candidate_agent_id"].endswith("qwen+adapter")
    assert captured["slice_task_count"] == 54
    assert captured["selected"] == [f"dev-{index}" for index in range(18, 54)]
