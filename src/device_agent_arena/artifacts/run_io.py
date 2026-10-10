from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from device_agent_arena.domain.models import TaskSpec
from device_agent_arena.harness.runner import Agent, EvaluationHarness, HarnessConfig

RUN_FORMAT_VERSION = "0.2"
GRADER_VERSION = "0.2"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def task_suite_payload(tasks: Iterable[TaskSpec]) -> bytes:
    return (
        "\n".join(json.dumps(asdict(task), ensure_ascii=False, sort_keys=True) for task in tasks)
        + "\n"
    ).encode("utf-8")


def git_provenance(workdir: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(workdir), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return {"commit": revision, "dirty": bool(status.strip())}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def source_tree_sha256() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(package_root.rglob("*.py")):
        digest.update(source.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_tree_sha256(path: str | Path) -> str:
    """Hash a model/checkpoint artifact by relative names and file bytes."""
    root = Path(path)
    if root.is_file():
        return hashlib.sha256(root.read_bytes()).hexdigest()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact path does not exist: {root}")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact directory is empty: {root}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _publish_exclusive(destination: Path, payload: bytes) -> None:
    """Publish complete bytes without overwriting a concurrently-created artifact."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())

        # A same-directory hard link is an atomic, fail-if-exists publish: the
        # destination can only point at the already-complete temporary inode.
        os.link(temporary_path, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite existing run artifact: {destination}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_tasks(
    *,
    tasks: Iterable[TaskSpec],
    agent_factory: Callable[[TaskSpec], Agent],
    agent_id: str,
    display_name: str,
    split: str,
    output: str | Path,
    config: HarnessConfig | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing run artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_config = config or HarnessConfig()
    task_list = tuple(tasks)
    task_payload = task_suite_payload(task_list)
    task_ids_payload = ("\n".join(task.id for task in task_list) + "\n").encode("utf-8")
    header = {
        "record_type": "run_header",
        "format_version": RUN_FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "agent_id": agent_id,
        "display_name": display_name,
        "split": split,
        "task_version": sorted({task.version for task in task_list}),
        "task_suite_sha256": _sha256(task_payload),
        "task_ids_sha256": _sha256(task_ids_payload),
        "grader_version": GRADER_VERSION,
        "harness_config": asdict(resolved_config),
        "task_count": len(task_list),
        "source": {
            **git_provenance(Path.cwd()),
            "tree_sha256": source_tree_sha256(),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "agent_config": run_metadata or {},
    }
    records: list[dict[str, Any]] = [header]
    harness = EvaluationHarness(resolved_config)
    for task in task_list:
        trial = harness.run(task, agent_factory(task))
        record = asdict(trial)
        record.update(
            {
                "record_type": "trial",
                "categories": list(task.categories),
                "seed": task.seed,
            }
        )
        records.append(record)
    payload = (
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
        + "\n"
    ).encode("utf-8")
    _publish_exclusive(
        destination,
        payload,
    )
    return destination


def load_run(path: str | Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    source = Path(path)
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if not records or records[0].get("record_type") != "run_header":
        raise ValueError("run artifact must begin with a run_header")
    trials = tuple(record for record in records[1:] if record.get("record_type") == "trial")
    if len(trials) != records[0].get("task_count"):
        raise ValueError("run artifact task_count does not match trial records")
    return records[0], trials
