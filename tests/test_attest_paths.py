# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
# A binding with a relative evidence path must load from any cwd (regression: invar serve --attest crashed
# on the KV260 because the relative path was resolved against the process cwd, not the binding's directory).
import json, os
from invar.attest import AttestationBinding


def test_relative_evidence_path_resolves_from_any_cwd(tmp_path, monkeypatch):
    d = tmp_path / "bundle"; d.mkdir()
    (d / "quote.msg").write_bytes(b"evidence-bytes")
    (d / "verdict.json").write_text('{"v":"ACCEPT"}')
    # bind with paths relative to the bundle dir, save the binding inside it
    monkeypatch.chdir(d)
    b = AttestationBinding("tpm-quote", "quote.msg", "tpm2_checkquote", "verdict.json")
    ev, vd, gen = b.evidence_digest, b.verdict_digest, b.genesis()
    b.save(str(d / "binding.json"))
    # load from a completely different cwd -> must still find the evidence and match digests
    monkeypatch.chdir(tmp_path)
    lb = AttestationBinding.load(str(d / "binding.json"))
    assert lb.evidence_digest == ev and lb.verdict_digest == vd
    assert lb.genesis() == gen


def test_missing_evidence_is_not_fatal_and_uses_saved_digest(tmp_path):
    d = tmp_path / "b2"; d.mkdir()
    (d / "quote.msg").write_bytes(b"xyz")
    b = AttestationBinding("tpm-quote", str(d / "quote.msg"))
    saved = d / "binding.json"; b.save(str(saved)); ev = b.evidence_digest
    (d / "quote.msg").unlink()                     # evidence truly gone
    lb = AttestationBinding.load(str(saved))        # must not raise
    assert lb.evidence_digest == ev                 # saved digest is authoritative
