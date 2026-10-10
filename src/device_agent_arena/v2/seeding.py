from __future__ import annotations

import hashlib
import re

_SECRET_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def external_test_root_seed(secret_hex: str, *, suite_namespace: str) -> int:
    """Derive a Test generator seed from an external 256-bit secret.

    The secret is deliberately not persisted in a public suite manifest.  Train/Dev
    seeds therefore cannot reconstruct the official Test split.
    """
    if _SECRET_PATTERN.fullmatch(secret_hex) is None:
        raise ValueError("locked Test requires a 64-hex-character external secret")
    payload = f"{suite_namespace}\0{secret_hex.lower()}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
