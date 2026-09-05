# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.scitt — worldline entries as SCITT-style Signed Statements (COSE_Sign1, RFC 9052).

A transparency log (SCITT, IETF) registers *signed statements* and answers "who
asserted what, when". OpenPCC's log holds software images; INVAR can put COMPUTATIONS
in one without leaking prompts, because the statement payload is the canonical CR
manifest — digests only. Layout (per the open-cr INTEROP.md §2 mapping):

  protected headers: alg (EdDSA -8 for the software key, ES256 -7 for the TPM ECDSA
                     P-256 key), content type application/vnd.anomly.cr+json,
                     CWT claims {iss: issuer, sub: the entry's certificate}
  payload:           canonical_bytes(manifest)   (canonical BEFORE enveloping)
  signature:         over the RFC 9052 Sig_structure ["Signature1", protected, "", payload]

The signature says WHO asserted; only CR re-execution says the claim is TRUE. No COSE
library: COSE_Sign1 is a small fixed CBOR structure, written out here so every byte on
the wire is inspectable (the demo in open-cr/examples/python is the reference).
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.asymmetric.utils import (decode_dss_signature,
                                                             encode_dss_signature)

from .crcore import canonical_bytes, certificate_of
from .hwsign import SOFTWARE, TPM2

CONTENT_TYPE = "application/vnd.anomly.cr+json"
ALG_EDDSA, ALG_ES256 = -8, -7
H_ALG, H_CTY, H_KID, H_CWT = 1, 3, 4, 15
CWT_ISS, CWT_SUB = 1, 2
COSE_SIGN1_TAG = 18


# ---------------------------------------------------------------- minimal CBOR

class Tag:
    def __init__(self, n: int, value):
        self.n, self.value = n, value


def _head(major: int, n: int) -> bytes:
    if n < 24:
        return bytes([(major << 5) | n])
    for bits, code, fmt in ((8, 24, "B"), (16, 25, ">H"), (32, 26, ">I"), (64, 27, ">Q")):
        if n < (1 << bits):
            return bytes([(major << 5) | code]) + struct.pack(fmt, n)
    raise ValueError("length too large")


def cbor(obj) -> bytes:
    if isinstance(obj, Tag):
        return _head(6, obj.n) + cbor(obj.value)
    if isinstance(obj, bool):
        raise TypeError("bool not used")
    if isinstance(obj, int):
        return _head(0, obj) if obj >= 0 else _head(1, -1 - obj)
    if isinstance(obj, bytes):
        return _head(2, len(obj)) + obj
    if isinstance(obj, str):
        b = obj.encode()
        return _head(3, len(b)) + b
    if isinstance(obj, list):
        return _head(4, len(obj)) + b"".join(cbor(x) for x in obj)
    if isinstance(obj, dict):
        return _head(5, len(obj)) + b"".join(cbor(k) + cbor(v) for k, v in obj.items())
    raise TypeError(type(obj))


def cbor_decode(b: bytes):
    ib, rest = b[0], b[1:]
    major, info = ib >> 5, ib & 0x1F
    if info < 24:
        n = info
    else:
        size = {24: 1, 25: 2, 26: 4, 27: 8}[info]
        n = int.from_bytes(rest[:size], "big")
        rest = rest[size:]
    if major == 0:
        return n, rest
    if major == 1:
        return -1 - n, rest
    if major == 2:
        return rest[:n], rest[n:]
    if major == 3:
        return rest[:n].decode(), rest[n:]
    if major == 4:
        out = []
        for _ in range(n):
            v, rest = cbor_decode(rest)
            out.append(v)
        return out, rest
    if major == 5:
        out = {}
        for _ in range(n):
            k, rest = cbor_decode(rest)
            v, rest = cbor_decode(rest)
            out[k] = v
        return out, rest
    if major == 6:
        v, rest = cbor_decode(rest)
        return Tag(n, v), rest
    raise ValueError(f"major {major} not supported")


# ---------------------------------------------------------------- signed statements

def _alg_for(signer) -> int:
    return {SOFTWARE: ALG_EDDSA, TPM2: ALG_ES256}[signer.backend]


def _cose_sig(signer, msg: bytes) -> bytes:
    """COSE signature bytes: raw 64-byte Ed25519, or raw r||s for ES256."""
    raw = signer.sign_raw(msg)
    if signer.backend == TPM2:
        return raw                          # TPM2Signer.sign_raw already returns r||s
    return raw


def signed_statement(entry: dict, signer, issuer: str) -> bytes:
    """COSE_Sign1 over the entry's canonical manifest; CWT sub = its certificate."""
    manifest = entry["manifest"]
    cert = entry["certificate"]
    if certificate_of(manifest) != cert:
        raise ValueError("entry certificate does not match its manifest")
    payload = canonical_bytes(manifest)
    protected = cbor({H_ALG: _alg_for(signer), H_CTY: CONTENT_TYPE,
                      H_KID: signer.key_id.encode(),
                      H_CWT: {CWT_ISS: issuer, CWT_SUB: cert}})
    sig_structure = cbor(["Signature1", protected, b"", payload])
    return cbor(Tag(COSE_SIGN1_TAG, [protected, {}, payload, _cose_sig(signer, sig_structure)]))


def verify_statement(stmt: bytes, pubkey_pem: str,
                     issuer: str | None = None) -> tuple[bool, dict]:
    """(ok, info). Checks tag, headers, signature (EdDSA or ES256), and that
    sha256(canonical(payload)) == CWT sub. Returns the decoded manifest in info."""
    info: dict = {}
    try:
        tag, _ = cbor_decode(stmt)
        if not isinstance(tag, Tag) or tag.n != COSE_SIGN1_TAG:
            return False, {"why": "not a tagged COSE_Sign1"}
        prot_b, _unprot, payload, sig = tag.value
        hdr, _ = cbor_decode(prot_b)
        alg, cty, cwt = hdr.get(H_ALG), hdr.get(H_CTY), hdr.get(H_CWT, {})
        info.update({"alg": alg, "cty": cty, "iss": cwt.get(CWT_ISS), "sub": cwt.get(CWT_SUB),
                     "kid": hdr.get(H_KID, b"").decode(errors="replace")})
        if cty != CONTENT_TYPE:
            return False, {**info, "why": "content type is not a CR manifest"}
        if issuer is not None and cwt.get(CWT_ISS) != issuer:
            return False, {**info, "why": "issuer mismatch"}
        pub = serialization.load_pem_public_key(pubkey_pem.encode())
        sig_structure = cbor(["Signature1", prot_b, b"", payload])
        try:
            if alg == ALG_EDDSA:
                pub.verify(sig, sig_structure)
            elif alg == ALG_ES256:
                r, s = int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big")
                pub.verify(encode_dss_signature(r, s), sig_structure, ec.ECDSA(hashes.SHA256()))
            else:
                return False, {**info, "why": f"unsupported alg {alg}"}
        except InvalidSignature:
            return False, {**info, "why": "signature invalid"}
        manifest = json.loads(payload)
        cert = certificate_of(manifest)
        if cert != cwt.get(CWT_SUB):
            return False, {**info, "why": "payload certificate != CWT subject"}
        info.update({"manifest": manifest, "certificate": cert,
                     "statement_sha256": hashlib.sha256(stmt).hexdigest()})
        return True, info
    except Exception as e:                              # malformed CBOR etc.
        return False, {**info, "why": f"malformed statement: {e}"}


def statements_for_worldline(path: str, signer, issuer: str,
                             indices: list[int] | None = None) -> list[bytes]:
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if indices is not None and i not in indices:
                continue
            out.append(signed_statement(json.loads(line), signer, issuer))
    return out


def b64(stmt: bytes) -> str:
    return base64.b64encode(stmt).decode()
