from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def freeze_contractgraph_candidate(
    *,
    project_root: str | Path,
    config_path: str | Path,
    adapter_path: str | Path,
    output_path: str | Path,
    evidence_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _inside(root, config_path)
    adapter = _inside(root, adapter_path)
    if not adapter.is_dir():
        raise ValueError("adapter_path must be a project-local directory")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    paths = {
        *(path for path in (root / "src").rglob("*.py")),
        *(path for path in (root / "tests").rglob("*.py")),
        *(path for path in (root / "contracts").glob("*.json")),
        config_file,
        _inside(root, str(config["dataset"])),
        _inside(root, str(config["manifest"])),
        *(path for path in adapter.rglob("*") if path.is_file()),
        *(_inside(root, path) for path in evidence_paths),
        *(root / name for name in ("pyproject.toml", "uv.lock", "AGENTS.md")),
    }
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"freeze inputs are missing: {missing}")
    file_hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    canonical = json.dumps(
        file_hashes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tree_hash = hashlib.sha256(canonical).hexdigest()
    receipt = {
        "status": "frozen",
        "protocol": "contractgraph-candidate-freeze-v0.3",
        "freeze_id": tree_hash[:16],
        "source_tree_sha256": tree_hash,
        "config": config_file.relative_to(root).as_posix(),
        "adapter": adapter.relative_to(root).as_posix(),
        "evidence": [
            _inside(root, path).relative_to(root).as_posix() for path in evidence_paths
        ],
        "file_count": len(file_hashes),
        "file_sha256": file_hashes,
    }
    destination = _inside(root, output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite freeze receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def verify_contractgraph_freeze(
    receipt_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    receipt_file = _inside(root, receipt_path)
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    mismatches: dict[str, dict[str, str | None]] = {}
    for relative, expected in receipt["file_sha256"].items():
        path = _inside(root, relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            mismatches[str(relative)] = {"expected": str(expected), "actual": actual}
    return {
        "status": "valid" if not mismatches else "changed",
        "freeze_id": receipt["freeze_id"],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _inside(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes project root: {value}")
    return path
