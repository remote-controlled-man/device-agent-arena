from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from device_agent_arena.v1.generate import generate_tasks
from device_agent_arena.v1.io import load_tasks, write_tasks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_suite_sha256(tasks: tuple) -> str:
    payload = "".join(
        json.dumps(asdict(task), ensure_ascii=False, sort_keys=True) + "\n"
        for task in tasks
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_suite(
    output_root: str | Path,
    *,
    root_seed: int,
    train_count: int,
    dev_count: int,
    test_count: int,
) -> dict:
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite v1 suite: {root}")
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, count in (("train", train_count), ("dev", dev_count)):
        tasks = generate_tasks(split, count=count, root_seed=root_seed)
        path = root / f"{split}.jsonl"
        write_tasks(path, tasks)
        files[split] = {
            "path": path.name,
            "record_count": count,
            "sha256": _sha256(path),
            "archetype_counts": dict(
                sorted(Counter(task.provenance["archetype"] for task in tasks).items())
            ),
        }
    generator_path = Path(__file__).with_name("generate.py")
    manifest = {
        "format_version": "deviceops-ul-suite-v1.0",
        "root_seed": root_seed,
        "files": files,
        "test": {
            "status": "locked",
            "record_count": test_count,
            "split_seed_base": 30_000,
            "unlock_policy": "dev-gate-first",
        },
        "generator_sha256": hashlib.sha256(generator_path.read_bytes()).hexdigest(),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validate_suite(root)
    return manifest


def validate_suite(output_root: str | Path) -> dict:
    root = Path(output_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != "deviceops-ul-suite-v1.0":
        raise ValueError("unsupported v1 suite format")
    test_status = manifest.get("test", {}).get("status")
    if test_status not in {"locked", "materialized"}:
        raise ValueError("v1 Test status must be locked or materialized")
    split_counts = {}
    markers = {}
    ids: set[str] = set()
    splits = ("train", "dev", "test") if test_status == "materialized" else ("train", "dev")
    for split in splits:
        metadata = manifest["test"] if split == "test" else manifest["files"][split]
        path = root / metadata["path"]
        if _sha256(path) != metadata["sha256"]:
            raise ValueError(f"{split} hash mismatch")
        tasks = load_tasks(path)
        if len(tasks) != metadata["record_count"]:
            raise ValueError(f"{split} record count mismatch")
        if any(task.split != split for task in tasks):
            raise ValueError(f"{split} contains a task from another split")
        split_ids = {task.id for task in tasks}
        if ids & split_ids:
            raise ValueError("v1 task IDs overlap across splits")
        ids.update(split_ids)
        markers[split] = {task.initial_world["secret_marker"] for task in tasks}
        split_counts[split] = len(tasks)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            if markers[left] & markers[right]:
                raise ValueError(f"v1 hidden entities overlap across {left} and {right}")
    if test_status == "locked" and (root / "test.jsonl").exists():
        raise ValueError("v1 Test was materialized before promotion")
    return {"valid": True, "split_counts": dict(sorted(split_counts.items()))}


def materialize_test(
    output_root: str | Path,
    *,
    model_gate_path: str | Path,
) -> dict:
    """Generate the frozen Test split once, and only after a passing paired Dev gate."""
    root = Path(output_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("test", {}).get("status")
    if status == "materialized" or (root / "test.jsonl").exists():
        raise FileExistsError("v1 Test is already materialized")
    validate_suite(root)
    gate_source = Path(model_gate_path)
    gate = json.loads(gate_source.read_text(encoding="utf-8"))
    if gate.get("promote_to_test") is not True:
        raise ValueError("Dev model gate did not authorize Test materialization")
    if gate.get("format_version") != "deviceops-ul-model-gate-v1.1":
        raise ValueError(
            "unsupported V1 model gate format; raw-run-bound v1.1 is required"
        )
    candidate_receipt = gate.get("candidate_run")
    baseline_receipt = gate.get("baseline_run")
    if not isinstance(candidate_receipt, dict) or not isinstance(
        baseline_receipt, dict
    ):
        raise ValueError("V1 model gate must contain candidate and baseline raw run receipts")

    def verified_run_path(label: str, receipt: dict) -> Path:
        raw_path = receipt.get("path")
        expected_hash = receipt.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"{label} raw run receipt is incomplete")
        path = Path(raw_path)
        if not path.is_absolute():
            path = (gate_source.parent / path).resolve()
        if not path.is_file():
            raise ValueError(f"{label} raw run is missing: {path}")
        if _sha256(path) != expected_hash:
            raise ValueError(f"{label} raw run hash mismatch")
        return path

    candidate_path = verified_run_path("candidate", candidate_receipt)
    baseline_path = verified_run_path("baseline", baseline_receipt)
    from device_agent_arena.v1.reporting import build_paired_model_gate

    bootstrap = gate.get("bootstrap", {})
    recomputed_gate = build_paired_model_gate(
        candidate_path=candidate_path,
        baseline_path=baseline_path,
        bootstrap_samples=int(bootstrap.get("samples", 0)),
        seed=int(bootstrap.get("seed", 0)),
    )
    if recomputed_gate != gate:
        raise ValueError("V1 model gate does not match its raw run evidence")
    dev_tasks = load_tasks(root / manifest["files"]["dev"]["path"])
    if gate.get("task_suite_sha256") != _task_suite_sha256(dev_tasks):
        raise ValueError("Dev model gate does not match the frozen Dev suite")

    test_config = manifest["test"]
    tasks = generate_tasks(
        "test",
        count=int(test_config["record_count"]),
        root_seed=int(manifest["root_seed"]),
    )
    existing_ids = {
        task.id
        for split in ("train", "dev")
        for task in load_tasks(root / manifest["files"][split]["path"])
    }
    if existing_ids & {task.id for task in tasks}:
        raise ValueError("generated Test IDs overlap with Train/Dev")
    test_path = root / "test.jsonl"
    write_tasks(test_path, tasks)
    manifest["test"] = {
        **test_config,
        "status": "materialized",
        "path": test_path.name,
        "sha256": _sha256(test_path),
        "archetype_counts": dict(
            sorted(Counter(task.provenance["archetype"] for task in tasks).items())
        ),
        "model_gate_path": gate_source.as_posix(),
        "model_gate_sha256": hashlib.sha256(gate_source.read_bytes()).hexdigest(),
        "generator_sha256_at_materialization": hashlib.sha256(
            Path(__file__).with_name("generate.py").read_bytes()
        ).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validate_suite(root)
    return manifest
