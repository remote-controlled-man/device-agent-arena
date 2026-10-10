from __future__ import annotations

import json
from pathlib import Path

import pytest

from device_agent_arena.v1.io import load_tasks
from device_agent_arena.v2.cli import _load_schema_suite_tasks
from device_agent_arena.v2.seeding import external_test_root_seed
from device_agent_arena.v2.suite import (
    profile_id_for_task,
    validate_schema_suite,
    write_schema_suite,
)


def test_v2_schema_suite_freezes_train_dev_without_precomputing_test_seed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v2"

    manifest = write_schema_suite(
        root,
        root_seed=20260901,
        train_count=21,
        dev_count=21,
        test_count=42,
    )

    assert manifest["format_version"] == "deviceops-ul-v2-schema-suite-v0.2"
    assert manifest["test"] == {
        "record_count": 42,
        "seed_policy": "independent-256-bit-external-secret-after-dev-gate",
        "status": "locked",
    }
    serialized = json.dumps(manifest)
    assert "test_seed" not in serialized
    assert "root_seed" not in manifest
    assert manifest["train_dev_seed"] == 20260901
    assert not (root / "test.jsonl").exists()
    assert validate_schema_suite(root)["valid"] is True
    profiles = [profile_id_for_task(task) for task in load_tasks(root / "dev.jsonl")]
    assert set(profiles) == {"identity", "schema-a", "schema-b"}


def test_locked_test_seed_requires_external_256_bit_secret() -> None:
    with pytest.raises(ValueError, match="64-hex"):
        external_test_root_seed("20260901", suite_namespace="schema-v0.2")
    assert external_test_root_seed("ab" * 32, suite_namespace="schema-v0.2") != 20260901


def test_v2_schema_suite_validation_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "v2"
    write_schema_suite(
        root,
        root_seed=20260901,
        train_count=7,
        dev_count=7,
        test_count=42,
    )
    (root / "dev.jsonl").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dev hash mismatch"):
        validate_schema_suite(root)


def test_v2_runner_loads_v2_suite_instead_of_v1_format(tmp_path: Path) -> None:
    root = tmp_path / "v2"
    write_schema_suite(
        root,
        root_seed=20260901,
        train_count=7,
        dev_count=7,
        test_count=42,
    )

    tasks = _load_schema_suite_tasks(root, "dev")

    assert len(tasks) == 7
    assert all(task.split == "dev" for task in tasks)
