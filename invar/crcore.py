# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.crcore — the three Computation-Receipts primitives INVAR needs, vendored
verbatim-equivalent from the open-cr reference (python/cr/receipt.py, Apache-2.0,
also Anomly) so the agent installs standalone. MUST stay byte-compatible with the
reference: the packaging smoke test compares both implementations on sample
manifests whenever OPEN_CR_PYTHON is available.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

_HASHES = {"sha256": hashlib.sha256, "sha512": hashlib.sha512}


class ReceiptError(ValueError):
    pass


def canonical_bytes(obj: Any) -> bytes:
    """Canonical UTF-8 JSON: sorted keys, tight separators, no NaN/Inf."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hash(alg: str):
    try:
        return _HASHES[alg]()
    except KeyError:
        raise ReceiptError(f"unsupported digest algorithm {alg!r}") from None


def digest_bytes(data: bytes, alg: str = "sha256") -> str:
    h = _hash(alg)
    h.update(data)
    return f"{alg}:{h.hexdigest()}"


def certificate_of(manifest: Mapping[str, Any]) -> str:
    """The certificate is the digest OF THE MANIFEST, binding model, input,
    computation and output together (spec: open-cr CR-v0.1)."""
    alg = manifest.get("digest_alg", "sha256")
    return digest_bytes(canonical_bytes(manifest), alg)
