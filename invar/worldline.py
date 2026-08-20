# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.worldline — receipt layer for SpaceTime OS local inference.

Every inference gets a WORLDLINE ENTRY: a CR-style receipt (canonical manifest ->
sha256 certificate, built with the open-cr reference machinery) binding
  { runtime binary digest, model file digest, prompt digest, decode params }
to { output digest }, chained hash-linked into worldline.jsonl.

HONEST PROFILE — "llamacpp-pinned-reexec-v0": this is a deployment-pinned
re-execution receipt (CR registered-profile style), NOT the exact-quire profile.
With temp=0, a fixed seed, and the same binary+model+thread count, llama.cpp
decode is reproducible on the same deployment, so the receipt is re-executable
evidence: verify() reruns the pinned computation and compares output digests.
Cross-machine bit-exactness is NOT claimed here — that is what the b-posit
exact-quire profile adds later. Say exactly this in any customer-facing copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time

from .crcore import canonical_bytes, certificate_of, digest_bytes  # noqa: F401
# (vendored verbatim-equivalent from open-cr; the packaging smoke cross-checks
# against the reference whenever OPEN_CR_PYTHON points at a checkout)

PROFILE = "llamacpp-pinned-reexec-v0"


def file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def run_inference(binary: str, model: str, prompt: str,
                  n_predict: int = 128, seed: int = 1, threads: int = 4) -> str:
    """Deterministically-pinned llama.cpp run; returns ONLY the generated text.

    Uses single-turn simple-io mode (-st --simple-io) — this llama.cpp line ships
    an interactive chat UI that blocks on stdin without -st. The UI echoes the
    prompt as a "> ..." line and appends a "[ Prompt: ... t/s ]" stats line whose
    numbers vary run-to-run, so the receipt digests the EXTRACTED generation only.
    """
    cmd = [binary, "-m", model, "-p", prompt, "-n", str(n_predict),
           "--temp", "0", "--seed", str(seed), "-t", str(threads),
           "-st", "--simple-io"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                         stdin=subprocess.DEVNULL)
    if out.returncode != 0:
        raise RuntimeError(f"llama.cpp failed: {out.stderr[-400:]}")
    text = out.stdout
    echo = "\n> " + prompt + "\n"
    start = text.rfind(echo)
    if start < 0:
        raise RuntimeError("could not locate prompt echo in llama.cpp output")
    gen = text[start + len(echo):]
    stats = gen.find("\n[ Prompt:")
    if stats >= 0:
        gen = gen[:stats]
    return gen.strip("\n")


def build_entry(binary: str, model: str, prompt: str, output: str,
                n_predict: int, seed: int, threads: int,
                prev_chain: str) -> dict:
    manifest = {
        "cr": "0.1",
        "profile": PROFILE,
        "computation": {
            "kind": "llm-decode",
            "runtime_digest": file_digest(binary),
            "model_digest": file_digest(model),
            "model_name": os.path.basename(model),
            "params": {"n_predict": n_predict, "seed": seed,
                       "threads": threads, "temp": 0},
        },
        "inputs": {"prompt": digest_bytes(prompt.encode())},
        "outputs": {"text": digest_bytes(output.encode())},
        "prev_chain": prev_chain,
        "unix_time": int(time.time()),
    }
    cert = certificate_of(manifest)
    chain = "sha256:" + hashlib.sha256(
        (prev_chain + cert).encode()).hexdigest()
    return {"manifest": manifest, "certificate": cert, "chain": chain}


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

    def infer_with_receipt(self, binary: str, model: str, prompt: str,
                           n_predict: int = 128, seed: int = 1,
                           threads: int = 4) -> tuple[str, dict]:
        output = run_inference(binary, model, prompt, n_predict, seed, threads)
        entry = build_entry(binary, model, prompt, output,
                            n_predict, seed, threads, self.tip)
        # stored beside, NOT under, the certificate (texts are evidence carried
        # with the receipt; the digests inside the manifest are what is certified)
        entry["prompt_text"] = prompt
        entry["output_text"] = output
        self.append(entry)
        return output, entry


def verify_worldline(path: str, binary: str, model: str,
                     prompts: dict[str, str], reexecute: bool = True) -> list:
    """Verify every entry: certificate matches its canonical manifest, the chain
    links, and (if reexecute) the pinned computation reproduces the output digest.
    `prompts` maps prompt digest -> prompt text for re-execution."""
    results = []
    prev = Worldline.GENESIS
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
                if pd not in prompts:
                    why = "structure ok (no prompt text for re-execution)"
                elif (file_digest(binary) != m["computation"]["runtime_digest"]
                      or file_digest(model) != m["computation"]["model_digest"]):
                    ok, why = False, "deployment differs (binary/model digest)"
                else:
                    p = m["computation"]["params"]
                    out = run_inference(binary, model, prompts[pd],
                                        p["n_predict"], p["seed"], p["threads"])
                    if digest_bytes(out.encode()) != m["outputs"]["text"]:
                        ok, why = False, "re-execution output digest differs"
                    else:
                        why = "re-executed, output digest matches"
            results.append((i, ok, why))
            prev = e["chain"]
    return results
