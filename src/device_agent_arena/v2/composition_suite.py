from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from device_agent_arena.v2.composition_generate import generate_composition_tasks
from device_agent_arena.v2.composition_io import (
    load_composition_tasks,
    write_composition_tasks,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_composition_suite(
    output_root: str | Path,
    *,
    root_seed: int,
    train_count: int,
    dev_count: int,
    test_count: int,
) -> dict:
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite V2 composition suite: {root}")
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, count in (("train", train_count), ("dev", dev_count)):
        tasks = generate_composition_tasks(split, count=count, root_seed=root_seed)
        path = root / f"{split}.jsonl"
        write_composition_tasks(path, tasks)
        files[split] = {
            "path": path.name,
            "record_count": len(tasks),
            "sha256": _sha256(path),
            "slice_counts": dict(sorted(Counter(task.slice_id for task in tasks).items())),
            "domain_counts": dict(sorted(Counter(task.domain for task in tasks).items())),
        }
    manifest = {
        "format_version": "deviceops-ul-v2-composition-suite-v0.2",
        "scope": "dual-control, composed infeasibility, and cross-domain dependency topology",
        "train_dev_seed": root_seed,
        "files": files,
        "test": {
            "status": "locked",
            "record_count": test_count,
            "seed_policy": "independent-256-bit-external-secret-after-full-v2-dev-gate",
        },
        "source_sha256": {
            "task_generator": _sha256(Path(__file__).with_name("composition_generate.py")),
            "task_model": _sha256(Path(__file__).with_name("composition_models.py")),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validate_composition_suite(root)
    return manifest


def validate_composition_suite(output_root: str | Path) -> dict:
    root = Path(output_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("format_version")
    if version not in {
        "deviceops-ul-v2-composition-suite-v0.1",
        "deviceops-ul-v2-composition-suite-v0.2",
    }:
        raise ValueError("unsupported V2 composition suite format")
    if version.endswith("v0.1"):
        amendment_path = root / "manifest.test-policy-v0.2.json"
        if not amendment_path.is_file():
            raise ValueError("legacy composition suite requires a Test-policy amendment")
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment.get("base_manifest_sha256") != _sha256(manifest_path):
            raise ValueError("composition Test-policy amendment base hash mismatch")
        test_policy = amendment.get("official_test", {})
    else:
        test_policy = manifest.get("test", {})
    if version.endswith("v0.2") and "root_seed" in manifest:
        raise ValueError("ambiguous root_seed can reconstruct Test; use train_dev_seed")
    if manifest.get("test", {}).get("status") != "locked":
        raise ValueError("V2 composition Test must remain locked")
    if "test_seed" in manifest.get("test", {}) or (root / "test.jsonl").exists():
        raise ValueError("V2 composition Test was exposed before the full Dev gate")
    if test_policy.get("seed_policy") != (
        "independent-256-bit-external-secret-after-full-v2-dev-gate"
    ):
        raise ValueError("V2 composition Test must use an independent external secret")
    ids: set[str] = set()
    markers: set[str] = set()
    split_counts = {}
    for split in ("train", "dev"):
        metadata = manifest["files"][split]
        path = root / metadata["path"]
        if _sha256(path) != metadata["sha256"]:
            raise ValueError(f"{split} hash mismatch")
        tasks = load_composition_tasks(path)
        if len(tasks) != metadata["record_count"]:
            raise ValueError(f"{split} record count mismatch")
        if any(task.split != split for task in tasks):
            raise ValueError(f"{split} contains a task from another split")
        split_ids = {task.id for task in tasks}
        if ids & split_ids:
            raise ValueError("composition task IDs overlap across splits")
        ids.update(split_ids)
        split_markers = {str(task.initial_world["secret_marker"]) for task in tasks}
        if markers & split_markers:
            raise ValueError("composition hidden entities overlap across splits")
        markers.update(split_markers)
        observed_slices = dict(sorted(Counter(task.slice_id for task in tasks).items()))
        observed_domains = dict(sorted(Counter(task.domain for task in tasks).items()))
        if observed_slices != metadata["slice_counts"]:
            raise ValueError(f"{split} slice counts mismatch")
        if observed_domains != metadata["domain_counts"]:
            raise ValueError(f"{split} domain counts mismatch")
        split_counts[split] = len(tasks)
    return {"valid": True, "split_counts": split_counts, "test_status": "locked"}
