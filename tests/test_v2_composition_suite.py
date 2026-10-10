from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from device_agent_arena.v2.composition_io import load_composition_tasks
from device_agent_arena.v2.composition_suite import (
    validate_composition_suite,
    write_composition_suite,
)


def test_composition_suite_balances_three_slices_and_domains_without_test_seed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "composition"

    manifest = write_composition_suite(
        root,
        root_seed=20260902,
        train_count=18,
        dev_count=18,
        test_count=36,
    )

    assert manifest["format_version"] == "deviceops-ul-v2-composition-suite-v0.2"
    assert manifest["test"] == {
        "record_count": 36,
        "seed_policy": "independent-256-bit-external-secret-after-full-v2-dev-gate",
        "status": "locked",
    }
    serialized = json.dumps(manifest)
    assert "test_seed" not in serialized
    assert "root_seed" not in manifest
    assert manifest["train_dev_seed"] == 20260902
    assert not (root / "test.jsonl").exists()
    assert validate_composition_suite(root)["valid"] is True

    dev = load_composition_tasks(root / "dev.jsonl")
    assert Counter(task.slice_id for task in dev) == {
        "composed_infeasible": 6,
        "cross_domain_chain": 6,
        "dual_control": 6,
    }
    assert Counter(task.domain for task in dev) == {
        "home": 6,
        "vehicle": 6,
        "workspace": 6,
    }
    assert {
        (task.slice_id, task.domain) for task in dev
    } == {
        (slice_id, domain)
        for slice_id in (
            "dual_control",
            "composed_infeasible",
            "cross_domain_chain",
        )
        for domain in ("home", "vehicle", "workspace")
    }
    topology_by_domain = {
        task.domain: (
            task.initial_world["metadata"]["topology"],
            len(task.initial_world["relations"]),
        )
        for task in dev
        if task.slice_id == "cross_domain_chain"
    }
    assert topology_by_domain == {
        "home": ("linear", 1),
        "vehicle": ("fan_out_join", 2),
        "workspace": ("diamond_gate", 3),
    }
    composed_variants = {
        task.domain: set() for task in dev if task.slice_id == "composed_infeasible"
    }
    for task in dev:
        if task.slice_id == "composed_infeasible":
            composed_variants[task.domain].add(task.provenance["variant"])
    assert composed_variants == {
        "home": {"feasible_control", "infeasible"},
        "vehicle": {"feasible_control", "infeasible"},
        "workspace": {"feasible_control", "infeasible"},
    }
    for domain in ("home", "vehicle", "workspace"):
        pair = [
            task
            for task in dev
            if task.slice_id == "composed_infeasible" and task.domain == domain
        ]
        assert len(pair) == 2
        assert len({task.provenance["scenario_group"] for task in pair}) == 1
        assert len({task.instruction for task in pair}) == 1
        assert len({task.public_context["target_id"] for task in pair}) == 1


def test_composition_suite_has_disjoint_ids_and_hidden_markers(tmp_path: Path) -> None:
    root = tmp_path / "composition"
    write_composition_suite(
        root,
        root_seed=20260902,
        train_count=9,
        dev_count=9,
        test_count=18,
    )

    train = load_composition_tasks(root / "train.jsonl")
    dev = load_composition_tasks(root / "dev.jsonl")

    assert {task.id for task in train}.isdisjoint(task.id for task in dev)
    assert {task.initial_world["secret_marker"] for task in train}.isdisjoint(
        task.initial_world["secret_marker"] for task in dev
    )


def test_composition_suite_rejects_tampering_and_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "composition"
    write_composition_suite(
        root,
        root_seed=20260902,
        train_count=9,
        dev_count=9,
        test_count=18,
    )

    with pytest.raises(FileExistsError):
        write_composition_suite(
            root,
            root_seed=20260902,
            train_count=9,
            dev_count=9,
            test_count=18,
        )

    (root / "dev.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dev hash mismatch"):
        validate_composition_suite(root)
