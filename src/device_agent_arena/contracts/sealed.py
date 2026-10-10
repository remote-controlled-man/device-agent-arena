from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from device_agent_arena.contracts.schema import load_contract_catalog
from device_agent_arena.training.verifier_data import (
    audit_verifier_records,
    build_domain_verifier_records,
)


def write_sealed_suite(
    blueprint_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Materialize a one-shot two-domain suite from a post-freeze blueprint."""
    blueprint_file = Path(blueprint_path)
    output = Path(output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite sealed suite: {output}")
    blueprint_bytes = blueprint_file.read_bytes()
    blueprint = json.loads(blueprint_bytes)
    if blueprint.get("format_version") != "0.3":
        raise ValueError("unsupported sealed blueprint format_version")
    domains = blueprint.get("domains")
    if not isinstance(domains, list) or len(domains) != 2:
        raise ValueError("sealed blueprint must contain exactly two domains")
    salt = blueprint.get("salt")
    if not isinstance(salt, str) or len(salt) < 16:
        raise ValueError("sealed blueprint salt must contain at least 16 characters")
    catalog_ids = [str(domain.get("catalog", {}).get("catalog_id")) for domain in domains]
    if len(set(catalog_ids)) != 2:
        raise ValueError("sealed catalog IDs must be unique")

    records: list[dict[str, Any]] = []
    rendered_catalogs: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="device-agent-sealed-") as temporary:
        temporary_root = Path(temporary)
        for domain in domains:
            catalog_payload = _bind_schema_digests(dict(domain["catalog"]))
            catalog_name = f"{catalog_payload['catalog_id']}.json"
            rendered = (
                json.dumps(
                    catalog_payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            temporary_catalog = temporary_root / catalog_name
            temporary_catalog.write_bytes(rendered)
            catalog = load_contract_catalog(temporary_catalog)
            domain_records = int(domain.get("records", 500))
            records.extend(
                build_domain_verifier_records(
                    split="test_xschema",
                    catalog=catalog,
                    templates=tuple(str(item) for item in domain["templates"]),
                    entities=tuple(str(item) for item in domain["entities"]),
                    requested_records=domain_records,
                    operation_id=str(domain["target_operation_id"]),
                    values=tuple(str(item) for item in domain["values"]),
                    states=tuple(domain["states"]),
                )
            )
            rendered_catalogs[catalog_name] = rendered

    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    ).encode("utf-8")
    audit = audit_verifier_records(records)
    if audit["hidden_runtime_fields"] != 0 or audit["label_replay_consistency"] != 1.0:
        raise ValueError("sealed suite audit failed")
    output.mkdir(parents=True, exist_ok=True)
    contract_output = output / "contracts"
    contract_output.mkdir()
    for name, payload in rendered_catalogs.items():
        (contract_output / name).write_bytes(payload)
    data_path = output / "test_xschema.jsonl"
    data_path.write_bytes(serialized)
    receipt = {
        "status": "sealed",
        "suite_id": str(blueprint["suite_id"]),
        "split": "test_xschema",
        "blueprint_sha256": hashlib.sha256(blueprint_bytes).hexdigest(),
        "salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
        "data_sha256": hashlib.sha256(serialized).hexdigest(),
        "catalog_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(rendered_catalogs.items())
        },
        "audit": audit,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def _bind_schema_digests(catalog: dict[str, Any]) -> dict[str, Any]:
    if catalog.get("format_version") != "1.0":
        raise ValueError("sealed catalog format_version must be 1.0")
    for operation in catalog.get("operations", []):
        tool = operation["tool"]
        canonical = json.dumps(
            tool,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        operation["schema_sha256"] = hashlib.sha256(canonical).hexdigest()
    return catalog
