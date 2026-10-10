"""Audit and export the curated public repository surface."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_BLOCKED_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
_BLOCKED_PARTS = {".git", ".hf-cache", ".pytest_cache", ".uv-cache", ".venv"}
_SENSITIVE_FILE_NAMES = {
    ".env",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
_SECRET_PATTERNS = (
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
_LOCAL_PATH_PATTERNS = (
    re.compile(rb"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    re.compile(rb"(?<!\\)\\\\[A-Za-z0-9._$-]+\\[A-Za-z0-9.$_-]+(?:\\|$)"),
    re.compile(
        rb"(?<![A-Za-z0-9])/(?:home|Users|tmp|var/tmp|private/tmp)(?:/|$)",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class ReleaseSpec:
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    max_file_bytes: int
    max_total_bytes: int


@dataclass(frozen=True)
class AuditIssue:
    code: str
    path: str
    detail: str


def load_spec(path: str | Path) -> ReleaseSpec:
    spec_path = Path(path)
    with spec_path.open("rb") as handle:
        payload = tomllib.load(handle)
    release = payload.get("release")
    if not isinstance(release, dict):
        raise ValueError("public release spec must contain a [release] table")
    includes = tuple(str(item) for item in release.get("include", ()))
    excludes = tuple(str(item) for item in release.get("exclude", ()))
    if "rename" in release:
        raise ValueError(
            "release.rename is not supported; public files must keep repository paths"
        )
    if not includes:
        raise ValueError("public release spec must include at least one path")
    max_file_bytes = int(release.get("max_file_bytes", 50_000_000))
    max_total_bytes = int(release.get("max_total_bytes", 100_000_000))
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise ValueError("public release size limits must be positive")
    return ReleaseSpec(
        includes=includes,
        excludes=excludes,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )


def _matches_any(relative_path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(relative_path, pattern) for pattern in patterns)


def collect_release_files(root: str | Path, spec: ReleaseSpec) -> list[Path]:
    repository = Path(root).resolve()
    selected: dict[str, Path] = {}
    for pattern in spec.includes:
        matches = [path for path in repository.glob(pattern) if path.is_file()]
        if not matches:
            raise FileNotFoundError(f"public release include matched no files: {pattern}")
        for path in matches:
            relative = path.relative_to(repository).as_posix()
            if not _matches_any(relative, spec.excludes):
                selected[relative] = path
    return [selected[key] for key in sorted(selected)]


def _relative_display(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def audit_release(
    root: str | Path,
    files: Sequence[Path],
    spec: ReleaseSpec,
) -> list[AuditIssue]:
    repository = Path(root).resolve()
    issues: list[AuditIssue] = []
    total_bytes = 0
    seen: set[str] = set()
    for source in files:
        path = Path(source)
        relative = _relative_display(repository, path)
        resolved = path.resolve()
        if relative in seen:
            issues.append(AuditIssue("duplicate-path", relative, "path selected more than once"))
            continue
        seen.add(relative)
        if not resolved.is_relative_to(repository):
            issues.append(AuditIssue("outside-root", relative, "resolved path leaves repository"))
            continue
        if path.is_symlink():
            issues.append(AuditIssue("symlink", relative, "symlinks are not exported"))
            continue
        if not path.is_file():
            issues.append(AuditIssue("missing-file", relative, "selected path is not a file"))
            continue
        if _BLOCKED_PARTS.intersection(Path(relative).parts):
            issues.append(AuditIssue("blocked-directory", relative, "local cache directory"))
        if path.suffix.lower() in _BLOCKED_SUFFIXES:
            issues.append(AuditIssue("model-weight", relative, "model weights belong outside Git"))
        lower_name = path.name.lower()
        if lower_name in _SENSITIVE_FILE_NAMES or lower_name.endswith((".key", ".p12", ".pem")):
            issues.append(AuditIssue("sensitive-filename", relative, "credential-like filename"))
        size = path.stat().st_size
        total_bytes += size
        if size > spec.max_file_bytes:
            issues.append(
                AuditIssue(
                    "oversized-file",
                    relative,
                    f"{size} bytes exceeds {spec.max_file_bytes}",
                )
            )
        content = path.read_bytes()
        if b"\x00" not in content:
            if any(pattern.search(content) for pattern in _LOCAL_PATH_PATTERNS):
                issues.append(
                    AuditIssue("local-absolute-path", relative, "contains a local absolute path")
                )
            if any(pattern.search(content) for pattern in _SECRET_PATTERNS):
                issues.append(
                    AuditIssue("possible-secret", relative, "contains a credential-like value")
                )
    if total_bytes > spec.max_total_bytes:
        issues.append(
            AuditIssue(
                "oversized-release",
                ".",
                f"{total_bytes} bytes exceeds {spec.max_total_bytes}",
            )
        )
    return sorted(issues, key=lambda issue: (issue.path, issue.code, issue.detail))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_release(
    root: str | Path,
    destination: str | Path,
    spec: ReleaseSpec,
) -> Path:
    repository = Path(root).resolve()
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"destination must not exist: {target}")
    files = collect_release_files(repository, spec)
    issues = audit_release(repository, files, spec)
    if issues:
        summary = "; ".join(f"{item.code}:{item.path}" for item in issues)
        raise ValueError(f"public release audit failed: {summary}")

    target.mkdir(parents=True)
    manifest_files: list[dict[str, object]] = []
    for source in files:
        relative = source.relative_to(repository)
        output = (target / relative).resolve()
        if not output.is_relative_to(target):
            raise ValueError(f"public release path leaves destination: {relative}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        manifest_files.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256(output),
                "size_bytes": output.stat().st_size,
            }
        )
    manifest = {
        "format_version": "device-agent-arena-public-release-v1",
        "files": manifest_files,
        "total_bytes": sum(int(item["size_bytes"]) for item in manifest_files),
    }
    manifest_path = target / "PUBLIC_RELEASE_MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit or export the curated DeviceAgent Arena public release."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        help="release spec path (default: <project-root>/public-release.toml)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="audit the curated public file set")
    export = subparsers.add_parser("export", help="copy the curated set to a new directory")
    export.add_argument("destination", type=Path)
    return parser


def entrypoint(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.project_root.resolve()
    spec_path = args.spec.resolve() if args.spec else root / "public-release.toml"
    spec = load_spec(spec_path)
    if args.command == "audit":
        files = collect_release_files(root, spec)
        issues = audit_release(root, files, spec)
        if issues:
            for issue in issues:
                print(f"{issue.code}: {issue.path}: {issue.detail}", file=sys.stderr)
            return 1
        total_bytes = sum(path.stat().st_size for path in files)
        print(f"public release audit passed: {len(files)} files, {total_bytes} bytes")
        return 0
    manifest_path = export_release(root, args.destination, spec)
    print(f"public release exported: {manifest_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
