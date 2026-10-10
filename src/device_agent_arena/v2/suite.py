from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from device_agent_arena.v1.generate import generate_tasks
from device_agent_arena.v1.io import load_tasks, write_tasks
from device_agent_arena.v1.models import V1TaskSpec

_SCHEMA_PROFILES = ("identity", "schema-a", "schema-b")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_id_for_task(task: V1TaskSpec) -> str:
    return _SCHEMA_PROFILES[task.seed % len(_SCHEMA_PROFILES)]


def write_schema_suite(
    output_root: str | Path,
    *,
    root_seed: int,
    train_count: int,
    dev_count: int,
    test_count: int,
) -> dict:
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite V2 schema suite: {root}")
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, count in (("train", train_count), ("dev", dev_count)):
        tasks = generate_tasks(split, count=count, root_seed=root_seed)
        path = root / f"{split}.jsonl"
        write_tasks(path, tasks)
        files[split] = {
            "path": path.name,
            "record_count": len(tasks),
            "sha256": _sha256(path),
            "archetype_counts": dict(
                sorted(Counter(task.provenance["archetype"] for task in tasks).items())
            ),
            "schema_profile_counts": dict(
                sorted(Counter(profile_id_for_task(task) for task in tasks).items())
            ),
        }
    manifest = {
        "format_version": "deviceops-ul-v2-schema-suite-v0.2",
        "scope": (
            "schema-shift and distractor slice over the stateful V1 task families; "
            "not yet the dual-control or cross-domain V2 suite"
        ),
        "train_dev_seed": root_seed,
        "files": files,
        "schema_profile_rule": "identity/schema-a/schema-b selected by task.seed modulo 3",
        "test": {
            "status": "locked",
            "record_count": test_count,
            "seed_policy": "independent-256-bit-external-secret-after-dev-gate",
        },
        "source_sha256": {
            "task_generator": _sha256(
                Path(__file__).parents[1] / "v1" / "generate.py"
            ),
            "schema_transform": _sha256(Path(__file__).with_name("schema_shift.py")),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validate_schema_suite(root)
    return manifest


def validate_schema_suite(output_root: str | Path) -> dict:
    root = Path(output_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("format_version")
    if version not in {
        "deviceops-ul-v2-schema-suite-v0.1",
        "deviceops-ul-v2-schema-suite-v0.2",
    }:
        raise ValueError("unsupported V2 schema suite format")
    if version.endswith("v0.1"):
        amendment_path = root / "manifest.test-policy-v0.2.json"
        if not amendment_path.is_file():
            raise ValueError("legacy V2 schema suite requires a Test-policy amendment")
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment.get("base_manifest_sha256") != _sha256(manifest_path):
            raise ValueError("V2 schema Test-policy amendment base hash mismatch")
        test_policy = amendment.get("official_test", {})
    else:
        test_policy = manifest.get("test", {})
    if version.endswith("v0.2") and "root_seed" in manifest:
        raise ValueError("ambiguous root_seed can reconstruct Test; use train_dev_seed")
    if manifest.get("test", {}).get("status") != "locked":
        raise ValueError("V2 schema Test must remain locked in this suite version")
    if "test_seed" in manifest.get("test", {}) or (root / "test.jsonl").exists():
        raise ValueError("V2 schema Test was exposed before the Dev gate")
    if test_policy.get("seed_policy") != (
        "independent-256-bit-external-secret-after-dev-gate"
    ):
        raise ValueError("V2 schema Test must use an independent external secret")
    split_counts = {}
    ids: set[str] = set()
    markers: set[str] = set()
    for split in ("train", "dev"):
        metadata = manifest["files"][split]
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
            raise ValueError("V2 task IDs overlap across splits")
        ids.update(split_ids)
        split_markers = {str(task.initial_world["secret_marker"]) for task in tasks}
        if markers & split_markers:
            raise ValueError("V2 hidden entities overlap across splits")
        markers.update(split_markers)
        expected_profiles = dict(
            sorted(Counter(profile_id_for_task(task) for task in tasks).items())
        )
        if expected_profiles != metadata["schema_profile_counts"]:
            raise ValueError(f"{split} schema profile counts mismatch")
        split_counts[split] = len(tasks)
    return {"valid": True, "split_counts": split_counts, "test_status": "locked"}
