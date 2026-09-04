# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.worldline — receipt layer for SpaceTime OS local inference.

Every inference gets a WORLDLINE ENTRY: a CR-style receipt (canonical manifest ->
sha256 certificate, built with the open-cr reference machinery) binding
  { runtime digest, model digest, prompt digest, decode params }
to { output digest }, chained hash-linked into worldline.jsonl.

The engine behind an entry is a BACKEND (invar.backends): llama.cpp or Ollama.
Each backend contributes its own deployment digests and its own profile string,
so a verifier knows exactly what it must re-run.

HONEST PROFILES — "llamacpp-pinned-reexec-v0" / "ollama-pinned-reexec-v0": these
are deployment-pinned re-execution receipts (CR registered-profile style), NOT the
exact-quire profile. With temp=0, a fixed seed, and the same runtime+model+params,
decode is reproducible on the same deployment, so the receipt is re-executable
evidence: verify() reruns the pinned computation and compares output digests.
Cross-machine bit-exactness is NOT claimed here — that is what the b-posit
exact-quire profile adds later. Say exactly this in any customer-facing copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

from .backends import (LLAMACPP_PROFILE, LlamaCppBackend, file_digest,  # noqa: F401
                       run_llamacpp)
from .crcore import canonical_bytes, certificate_of, digest_bytes  # noqa: F401
# (vendored verbatim-equivalent from open-cr; the packaging smoke cross-checks
# against the reference whenever OPEN_CR_PYTHON points at a checkout)

PROFILE = LLAMACPP_PROFILE          # the original/default profile name
run_inference = run_llamacpp        # back-compat alias (tests, external callers)

# deployment keys a verifier compares against the live backend; anything else
# under `computation` (model_name, runtime_version, runtime_pinned_by) is
# descriptive and certified but not a re-execution precondition
PIN_KEYS = ("runtime_digest", "model_digest", "weights_digest")


def build_entry_for(backend, prompt: str, output: str, params: dict,
                    prev_chain: str, deployment: dict | None = None) -> dict:
    dep = deployment if deployment is not None else backend.deployment()
    manifest = {
        "cr": "0.1",
        "profile": backend.profile,
        "computation": {"kind": "llm-decode", **dep, "params": params},
        "inputs": {"prompt": digest_bytes(prompt.encode())},
        "outputs": {"text": digest_bytes(output.encode())},
        "prev_chain": prev_chain,
        "unix_time": int(time.time()),
    }
    cert = certificate_of(manifest)
    chain = "sha256:" + hashlib.sha256(
        (prev_chain + cert).encode()).hexdigest()
    return {"manifest": manifest, "certificate": cert, "chain": chain}


def build_entry(binary: str, model: str, prompt: str, output: str,
                n_predict: int, seed: int, threads: int,
                prev_chain: str) -> dict:
    """llama.cpp entry (original signature; manifest layout unchanged)."""
    b = LlamaCppBackend(binary, model, threads)
    return build_entry_for(b, prompt, output, b.params(n_predict, seed),
                           prev_chain)


class Worldline:
    """Append-only hash-linked receipt log; one file per deployment."""

    GENESIS = "sha256:" + "0" * 64

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.tip = self.GENESIS
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    self.tip = json.loads(line)["chain"]

    def append(self, entry: dict) -> None:
        with self._lock:
            assert entry["manifest"]["prev_chain"] == self.tip, "chain fork"
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, separators=(",", ":"),
                                   sort_keys=True) + "\n")
            self.tip = entry["chain"]

    def infer(self, backend, prompt: str, n_predict: int = 128,
              seed: int = 1, deployment: dict | None = None) -> tuple[str, dict]:
        """Run the pinned computation on `backend`, receipt it, append it.
        `deployment` may be passed to reuse digests computed once at startup
        (hashing a multi-GB gguf per request would be silly; the digests are
        re-read at verify time regardless)."""
        params = backend.params(n_predict, seed)
        output = backend.generate(prompt, params)
        entry = build_entry_for(backend, prompt, output, params, self.tip,
                                deployment)
        # stored beside, NOT under, the certificate (texts are evidence carried
        # with the receipt; the digests inside the manifest are what is certified)
        entry["prompt_text"] = prompt
        entry["output_text"] = output
        self.append(entry)
        return output, entry

    def infer_with_receipt(self, binary: str, model: str, prompt: str,
                           n_predict: int = 128, seed: int = 1,
                           threads: int = 4) -> tuple[str, dict]:
        """llama.cpp convenience (original signature)."""
        return self.infer(LlamaCppBackend(binary, model, threads), prompt,
                          n_predict, seed)


def verify_entries(path: str, prompts: dict[str, str], backends: dict,
                   reexecute: bool = True) -> list:
    """Verify every entry: certificate matches its canonical manifest, the chain
    links, and (if reexecute) the pinned computation reproduces the output digest.
    `prompts` maps prompt digest -> prompt text for re-execution.
    `backends` maps profile -> backend instance, or -> factory(model_name) that
    builds one per model named in the receipts (a worldline may mix profiles and
    models; an entry whose profile has no backend gets structural checks only)."""
    results = []
    prev = Worldline.GENESIS
    live: dict[str, dict] = {}       # (profile, model) -> deployment(), once
    inst: dict[tuple, object] = {}   # factory results, keyed the same way

    def _backend(profile: str, comp: dict):
        be = backends.get(profile)
        if be is None or hasattr(be, "profile"):
            return be                       # None or a ready backend instance
        key = (profile, comp.get("model_name", ""))
        if key not in inst:                 # factory(model_name) -> backend
            inst[key] = be(comp.get("model_name", ""))
        return inst[key]
    with open(path) as f:
        for i, line in enumerate(f):
            e = json.loads(line)
            m, ok, why = e["manifest"], True, "ok"
            if certificate_of(m) != e["certificate"]:
                ok, why = False, "certificate mismatch"
            elif m["prev_chain"] != prev:
                ok, why = False, "chain broken"
            elif e["chain"] != "sha256:" + hashlib.sha256(
                    (prev + e["certificate"]).encode()).hexdigest():
                ok, why = False, "chain digest wrong"
            elif reexecute:
                pd = m["inputs"]["prompt"]
                be = _backend(m.get("profile"), m["computation"])
                if be is None:
                    why = (f"structure ok (no backend for profile "
                           f"{m.get('profile')!r}; not re-executed)")
                elif pd not in prompts:
                    why = "structure ok (no prompt text for re-execution)"
                else:
                    comp = m["computation"]
                    key = (be.profile, comp.get("model_name", ""))
                    if key not in live:
                        live[key] = be.deployment()
                    dep = live[key]
                    diff = [k for k in PIN_KEYS
                            if k in comp and comp[k] != dep.get(k)]
                    if diff:
                        ok, why = False, ("deployment differs "
                                          f"({', '.join(diff)})")
                    else:
                        out = be.generate(prompts[pd], comp["params"])
                        if digest_bytes(out.encode()) != m["outputs"]["text"]:
                            ok, why = False, "re-execution output digest differs"
                        else:
                            why = "re-executed, output digest matches"
            results.append((i, ok, why))
            prev = e["chain"]
    return results


def verify_worldline(path: str, binary: str, model: str,
                     prompts: dict[str, str], reexecute: bool = True) -> list:
    """llama.cpp verification (original signature)."""
    backends = {}
    if reexecute and binary and model:
        backends[LLAMACPP_PROFILE] = LlamaCppBackend(binary, model)
    return verify_entries(path, prompts, backends, reexecute)
