from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path

import pytest

from tools.public_release import (
    ReleaseSpec,
    audit_release,
    collect_release_files,
    export_release,
    load_spec,
)

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def test_repository_public_release_is_focused_and_auditable() -> None:
    spec = load_spec(ROOT / "public-release.toml")
    files = collect_release_files(ROOT, spec)
    relative = {path.relative_to(ROOT).as_posix() for path in files}

    assert "README.md" in relative
    assert "README.zh-CN.md" in relative
    assert "LICENSE" in relative
    assert "docs/V2_7_LAYERED_EXPERIMENT_REPORT.zh-CN.md" in relative
    assert "artifacts/v2/layered/release-manifest-v2.7.json" in relative
    assert "tests/test_v2_layered_agent.py" in relative
    assert "tests/test_v2_layered_release_evidence.py" not in relative
    assert not any(path.endswith(".safetensors") for path in relative)
    assert not any("/.hf-cache/" in f"/{path}/" for path in relative)
    assert not any(path.endswith("/test.jsonl") for path in relative)
    assert audit_release(ROOT, files, spec) == []


def test_public_readme_links_resolve_inside_the_export() -> None:
    spec = load_spec(ROOT / "public-release.toml")
    files = collect_release_files(ROOT, spec)
    destinations = {
        path.relative_to(ROOT).as_posix()
        for path in files
    }

    for source_name in ("README.md", "README.zh-CN.md"):
        destination_name = source_name
        content = (ROOT / source_name).read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(content):
            link = match.group(1).split("#", 1)[0]
            if not link or "://" in link or link.startswith(("#", "mailto:")):
                continue
            target = posixpath.normpath(
                posixpath.join(posixpath.dirname(destination_name), link)
            )
            assert target in destinations, f"broken public README link: {source_name} -> {link}"


def test_audit_rejects_secrets_and_local_absolute_paths(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "token = " + "ghp_" + "abcdefghijklmnopqrstuvwxyz123456\n"
        "checkpoint = " + "E:" + "\\career\\private\\model\n",
        encoding="utf-8",
    )
    spec = ReleaseSpec(
        includes=("README.md",),
        excludes=(),
        max_file_bytes=1_000_000,
        max_total_bytes=1_000_000,
    )

    issues = audit_release(tmp_path, [readme], spec)
    codes = {issue.code for issue in issues}

    assert "possible-secret" in codes
    assert "local-absolute-path" in codes


@pytest.mark.parametrize(
    "local_path",
    (
        "D:" + "\\models\\checkpoint",
        "D:" + "/models/checkpoint",
        "\\" * 2 + "build-server\\model-share\\checkpoint",
        "/" + "home/researcher/checkpoint",
        "/" + "Users/researcher/checkpoint",
        "/" + "tmp/checkpoint",
    ),
)
def test_audit_rejects_common_absolute_path_styles(
    tmp_path: Path, local_path: str
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"checkpoint = {local_path}\n", encoding="utf-8")
    spec = ReleaseSpec(
        includes=("README.md",),
        excludes=(),
        max_file_bytes=1_000_000,
        max_total_bytes=1_000_000,
    )

    issues = audit_release(tmp_path, [readme], spec)

    assert "local-absolute-path" in {issue.code for issue in issues}


def test_load_spec_rejects_path_renaming(tmp_path: Path) -> None:
    spec_path = tmp_path / "public-release.toml"
    spec_path.write_text(
        '[release]\ninclude = ["README.md"]\n\n'
        '[release.rename]\n"README.md" = "docs/README.md"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rename is not supported"):
        load_spec(spec_path)


def test_export_writes_hash_manifest_and_refuses_nonempty_destination(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir(parents=True)
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    spec = ReleaseSpec(
        includes=("README.md",),
        excludes=(),
        max_file_bytes=1_000_000,
        max_total_bytes=1_000_000,
    )
    destination = tmp_path / "public"

    manifest_path = export_release(root, destination, spec)
    assert b"\r\n" not in manifest_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format_version"] == "device-agent-arena-public-release-v1"
    assert manifest["files"][0]["path"] == "README.md"
    assert len(manifest["files"][0]["sha256"]) == 64
    with pytest.raises(FileExistsError, match="destination must not exist"):
        export_release(root, destination, spec)
