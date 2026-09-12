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

from .backends import (LLAMACPP_PROFILE, LLAMACPP_EXACT_PROFILE, UPSTREAM_PINNED_PROFILE, LlamaCppBackend, file_digest,  # noqa: F401
                       run_llamacpp)
from .attest import NONE as ATTEST_NONE, AttestationBinding, check_binding  # noqa: F401
from .hwsign import verify_signature
from .crcore import canonical_bytes, certificate_of, digest_bytes  # noqa: F401
# (vendored verbatim-equivalent from open-cr; the packaging smoke cross-checks
# against the reference whenever OPEN_CR_PYTHON points at a checkout)

PROFILE = LLAMACPP_PROFILE          # the original/default profile name
run_inference = run_llamacpp        # back-compat alias (tests, external callers)

# deployment keys a verifier compares against the live backend; anything else
# under `computation` (model_name, runtime_version, runtime_pinned_by) is
# descriptive and certified but not a re-execution precondition
PIN_KEYS = ("runtime_digest", "model_digest", "weights_digest", "device", "n_gpu_layers")


def build_entry_for(backend, prompt: str, output: str, params: dict,
                    prev_chain: str, deployment: dict | None = None,
                    host_attestation: dict | None = None) -> dict:
    dep = deployment if deployment is not None else backend.deployment()
    comp = {"kind": "llm-decode", **dep, "params": params}
    if host_attestation and host_attestation.get("kind", "none") != "none":
        comp["host_attestation"] = host_attestation   # certified: cannot be re-homed
    sc = getattr(backend, "spot_check_field", None)
    if sc is not None:
        field = sc()
        if field:
            comp["spot_check"] = field                # certified dump digest (CSC)
    manifest = {
        "cr": "0.1",
        "profile": backend.profile,
        "computation": comp,
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
    """Append-only hash-linked receipt log; one file per deployment.

    `binding`  (invar.attest.AttestationBinding) — when given, genesis is derived
               from the platform's attestation evidence and every manifest carries
               `host_attestation`; the chain is then unusable under any other evidence.
    `signer`   (invar.hwsign.*Signer) — when given, every appended entry gets a
               `signature` block over its chain digest (TPM-resident key or software
               key, stated in the block)."""

    GENESIS = "sha256:" + "0" * 64

    def __init__(self, path: str, signer=None, binding=None):
        self.path = path
        self._lock = threading.Lock()
        self.signer = signer
        self.binding = binding
        self.genesis = binding.genesis() if binding else self.GENESIS
        self.tip = self.genesis
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    self.tip = json.loads(line)["chain"]

    @property
    def host_attestation(self) -> dict:
        return self.binding.manifest_field() if self.binding else dict(ATTEST_NONE)

    def append(self, entry: dict) -> None:
        with self._lock:
            assert entry["manifest"]["prev_chain"] == self.tip, "chain fork"
            if self.signer is not None:
                entry["signature"] = self.signer.sign(entry)
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, separators=(",", ":"),
                                   sort_keys=True) + "\n")
            self.tip = entry["chain"]

    def infer(self, backend, prompt: str, n_predict: int = 128,
              seed: int = 1, deployment: dict | None = None,
              chat: str | None = None) -> tuple[str, dict]:
        """Run the pinned computation on `backend`, receipt it, append it.
        `deployment` may be passed to reuse digests computed once at startup
        (hashing a multi-GB gguf per request would be silly; the digests are
        re-read at verify time regardless)."""
        params = backend.params(n_predict, seed)
        if chat:
            params["chat"] = chat              # certified: verify re-executes in the same mode
        output = backend.generate(prompt, params)
        entry = build_entry_for(backend, prompt, output, params, self.tip,
                                deployment, self.host_attestation)
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


def text_copies_mismatch(e: dict) -> str:
    """The certificate covers digests; `prompt_text` / `output_text` are readable copies
    written beside them. A copy that does not hash to its certified digest is a lie a
    reader would believe (the certificate still verifies), so it is a REJECT reason, not a
    warning. Returns the reason or ""."""
    m = e.get("manifest") or {}
    if "output_text" in e and digest_bytes(str(e["output_text"]).encode()) != (m.get("outputs") or {}).get("text"):
        return "output_text does not match the certified output digest"
    if "prompt_text" in e and digest_bytes(str(e["prompt_text"]).encode()) != (m.get("inputs") or {}).get("prompt"):
        return "prompt_text does not match the certified prompt digest"
    return ""


def verify_entries(path: str, prompts: dict[str, str], backends: dict,
                   reexecute: bool = True, binding=None,
                   trusted_key_ids: set[str] | None = None,
                   require_signature: bool = False,
                   cross_deployment: bool = False,
                   start_prev: str | None = None) -> list:
    """Verify every entry: certificate matches its canonical manifest, the chain
    links, and (if reexecute) the pinned computation reproduces the output digest.
    Each result is (index, ok, why) with ok True (ACCEPT), False (REJECT) or None
    (INDETERMINATE: receipt intact, but the deployment could not reproduce its own
    output on two fresh replays, so re-execution proves nothing either way).
    `prompts` maps prompt digest -> prompt text for re-execution.
    `backends` maps profile -> backend instance, or -> factory(model_name) that
    builds one per model named in the receipts (a worldline may mix profiles and
    models; an entry whose profile has no backend gets structural checks only).
    `cross_deployment`: for EXACT-profile entries, re-execute even when the certified
    deployment pins (runtime, device, n_gpu_layers) differ from the verifier's — the
    exact profile's whole graph is deterministic across CPU and CUDA, so the output
    digest must still match; the differing pins are reported in the reason. Float
    profiles keep the pin regardless."""
    results = []
    # start_prev: verify a contiguous TAIL of a worldline (e.g. fetched from /v1/worldline/tail): the
    # first entry links to the given chain digest instead of the genesis.
    prev = binding.genesis() if binding else (start_prev if start_prev is not None else Worldline.GENESIS)
    live: dict[str, dict] = {}       # (profile, model) -> deployment(), once
    inst: dict[tuple, object] = {}   # factory results, keyed the same way
    # The log can testify against its own deployment: two entries with the same request
    # (profile, model, prompt digest, params) and different certified outputs mean the
    # deployment is not reproducible, whatever a replay says today. Index that first.
    flaky: dict[tuple, list] = {}
    seen_req: dict[tuple, tuple] = {}
    try:
        with open(path) as f0:
            for i0, line0 in enumerate(f0):
                e0 = json.loads(line0); m0 = e0.get("manifest") or {}; c0 = m0.get("computation") or {}
                k = (m0.get("profile"), c0.get("model_name"), (m0.get("inputs") or {}).get("prompt"),
                     json.dumps(c0.get("params"), sort_keys=True))
                out0 = (m0.get("outputs") or {}).get("text")
                if k in seen_req and seen_req[k][1] != out0:
                    flaky.setdefault((k[0], k[1]), []).append((seen_req[k][0], i0))
                seen_req.setdefault(k, (i0, out0))
    except (OSError, ValueError):
        pass

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
            elif (bad := text_copies_mismatch(e)):
                ok, why = False, bad
            elif m["prev_chain"] != prev:
                ok, why = False, "chain broken"
            elif e["chain"] != "sha256:" + hashlib.sha256(
                    (prev + e["certificate"]).encode()).hexdigest():
                ok, why = False, "chain digest wrong"
            elif binding is not None and not check_binding(m, binding)[0]:
                ok, why = False, check_binding(m, binding)[1]
            elif (require_signature or e.get("signature")) and not (
                    sig_ok := verify_signature(e, trusted_key_ids))[0]:
                ok, why = False, sig_ok[1]
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
                    crossing = bool(diff) and cross_deployment and \
                        m.get("profile") == LLAMACPP_EXACT_PROFILE
                    if diff and not crossing:
                        ok, why = False, ("deployment differs "
                                          f"({', '.join(diff)})")
                    else:
                        out = be.generate(prompts[pd], comp["params"])
                        if digest_bytes(out.encode()) != m["outputs"]["text"]:
                            # Before calling it tampering, ask the deployment once more. A float
                            # upstream (vLLM, SGLang, TGI) that gives two different answers to the
                            # same greedy request cannot re-execute anything: the receipt is intact,
                            # the computation is simply not reproducible there. That is a third
                            # verdict, not a rejection, and it must not read like tampering.
                            upstream = m.get("profile") == UPSTREAM_PINNED_PROFILE
                            replays = {digest_bytes(out.encode())}
                            if upstream:
                                for _ in range(2):
                                    replays.add(digest_bytes(be.generate(prompts[pd], comp["params"]).encode()))
                            pairs = flaky.get((m.get("profile"), comp.get("model_name")), []) if upstream else []
                            if len(replays) > 1:
                                ok, why = None, ("INDETERMINATE: the upstream did not reproduce its own output "
                                                 f"({len(replays)} different answers in 3 fresh greedy replays); certificate and "
                                                 "chain intact; this deployment cannot re-execute the entry -- witness-grade "
                                                 "provenance, or use the exact tier")
                            elif pairs:
                                ok, why = None, ("INDETERMINATE: fresh replays agree with each other but not with the certificate, "
                                                 "and this log already shows the deployment answering identical requests differently "
                                                 f"(entries {', '.join(f'{a}/{b}' for a, b in pairs[:3])}); a non-reproducible deployment "
                                                 "cannot confirm or refute this entry -- witness-grade provenance, or use the exact tier")
                            else:
                                ok, why = False, ("re-execution output digest differs"
                                                  + (" (three fresh replays agree with each other, not with the certificate, and "
                                                     "this log shows no identical request answered differently)" if upstream else "")
                                                  + (f" (cross-deployment: {', '.join(diff)} differ)" if crossing else ""))
                        elif crossing:
                            why = ("re-executed CROSS-DEPLOYMENT, output digest matches "
                                   f"(certified {', '.join(diff)} differ from this verifier's)")
                        else:
                            why = "re-executed, output digest matches"
            if ok and binding is None and m.get("computation", {}).get(
                    "host_attestation", {}).get("kind", "none") != "none":
                why += ("; claims host attestation "
                        f"{m['computation']['host_attestation']['kind']} (NOT checked: pass --attest)")
            if ok and e.get("signature"):
                why += "; " + verify_signature(e, trusted_key_ids)[1]
            if ok and binding is not None:
                why += "; " + check_binding(m, binding)[1]
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
