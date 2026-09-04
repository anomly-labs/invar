# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.attest — bind a worldline to the platform's attestation evidence.

What a TEE / TPM attestation proves: which hardware, firmware, VM image and (with
OpenPCC-style PCR 12) which weight file booted. What it does not prove: what was
computed. A worldline proves what was computed. Binding the two gives one object a
client can check for both. Design in docs/research/verifiable-confidential-ai-2026-09-04.md §4.1.

INVAR does NOT re-implement SEV-SNP / TDX / NVIDIA-ETA verification. Those are the
vendors' and the Confidential Computing Consortium's business (snpguest, tdx-guest,
nvtrust / NRAS, Veraison, Intel Trust Authority). INVAR takes the evidence bytes and the
external verifier's signed verdict as inputs, records their digests, and binds the
chain to them:

  genesis = "sha256:" + sha256( "invar-genesis-v1" || evidence_digest || nonce )

so the FIRST entry's prev_chain already commits to the platform state, and every
manifest carries `host_attestation` = {kind, evidence_digest, verifier, verdict_digest,
nonce}, or {kind: "none"} when the host has nothing to attest with. A receipt cannot be
re-homed onto a different attestation without breaking every certificate after genesis.

Evidence kinds (free-form, but use these names): "sev-snp-report", "tdx-quote",
"nvidia-eta", "tpm-quote", "openpcc-bundle", "none".
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets

NONE = {"kind": "none"}


def digest_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class AttestationBinding:
    """Immutable description of the platform evidence a worldline is bound to."""

    def __init__(self, kind: str, evidence_path: str | None = None,
                 verifier: str | None = None, verdict_path: str | None = None,
                 nonce: str | None = None):
        self.kind = kind
        self.evidence_path = evidence_path
        self.verifier = verifier
        self.verdict_path = verdict_path
        self.nonce = nonce or secrets.token_hex(16)
        self.evidence_digest = digest_file(evidence_path) if evidence_path else None
        self.verdict_digest = digest_file(verdict_path) if verdict_path else None

    @classmethod
    def none(cls) -> "AttestationBinding":
        b = cls("none", nonce="0" * 32)
        return b

    def manifest_field(self) -> dict:
        if self.kind == "none":
            return dict(NONE)
        d = {"kind": self.kind, "evidence_digest": self.evidence_digest,
             "nonce": self.nonce}
        if self.verifier:
            d["verifier"] = self.verifier
        if self.verdict_digest:
            d["verdict_digest"] = self.verdict_digest
        return d

    def genesis(self) -> str:
        if self.kind == "none":
            return "sha256:" + "0" * 64            # the classic unbound genesis
        h = hashlib.sha256(b"invar-genesis-v1")
        h.update((self.evidence_digest or "").encode())
        h.update(self.nonce.encode())
        return "sha256:" + h.hexdigest()

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"kind": self.kind, "evidence_path": self.evidence_path,
                       "verifier": self.verifier, "verdict_path": self.verdict_path,
                       "nonce": self.nonce, "evidence_digest": self.evidence_digest,
                       "verdict_digest": self.verdict_digest,
                       "genesis": self.genesis()}, f, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "AttestationBinding":
        with open(path) as f:
            d = json.load(f)
        b = cls(d["kind"], d.get("evidence_path"), d.get("verifier"),
                d.get("verdict_path"), d.get("nonce"))
        # the saved digests are authoritative if the evidence file moved
        if b.evidence_digest is None:
            b.evidence_digest = d.get("evidence_digest")
        if b.verdict_digest is None:
            b.verdict_digest = d.get("verdict_digest")
        return b


def check_binding(entry_manifest: dict, binding: AttestationBinding) -> tuple[bool, str]:
    """Does this entry claim the same platform evidence the verifier holds?"""
    got = entry_manifest.get("computation", {}).get("host_attestation") \
        or entry_manifest.get("host_attestation") or NONE
    want = binding.manifest_field()
    if got.get("kind") == "none" and want.get("kind") == "none":
        return True, "no host attestation (stated)"
    if got == want:
        return True, f"host attestation bound ({got['kind']}, evidence {got['evidence_digest'][:23]}…)"
    return False, "host attestation differs from verifier's evidence"


def binding_from_env() -> AttestationBinding | None:
    """INVAR_ATTEST=<binding.json> to serve under a saved binding."""
    p = os.environ.get("INVAR_ATTEST")
    return AttestationBinding.load(p) if p else None


# --------------------------------------------------------------------------- evidence collectors

PCR_SYSFS = "/sys/class/tpm/tpm0/pcr-sha256"


def collect_pcr_bank(out_path: str, pcrs: tuple[int, ...] = tuple(range(0, 8)),
                     sysfs: str = PCR_SYSFS) -> dict:
    """Snapshot the TPM's SHA-256 PCR bank from sysfs (world-readable on Linux >= 5.12,
    no tss group needed). This is REAL measured-boot state but UNSIGNED — a TPM quote
    (tpm2_quote, needs device access) is the signed form; kind it "tpm-pcr-bank" so a
    verifier knows which it got."""
    vals = {}
    for i in pcrs:
        with open(os.path.join(sysfs, str(i))) as f:
            vals[str(i)] = f.read().strip().lower()
    doc = {"kind": "tpm-pcr-bank", "bank": "sha256", "pcrs": vals,
           "source": sysfs, "signed": False}
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    return doc
