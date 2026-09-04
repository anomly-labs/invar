# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.backends — the inference engines a worldline can pin.

A backend answers three questions and nothing else:
  deployment()          what is running: digests that identify the runtime and the
                        weights (these go INTO the certified manifest)
  params(n_predict,seed) the decode parameters that must be replayed verbatim
  generate(prompt,params) run the pinned computation, return ONLY the generated text

Two backends ship:

  LlamaCppBackend  profile "llamacpp-pinned-reexec-v0"
      pins: sha256(llama-cli binary), sha256(gguf file)
  OllamaBackend    profile "ollama-pinned-reexec-v0"
      pins: sha256(ollama binary) when it is resolvable locally (else the server's
            reported version, a weaker pin — and the receipt SAYS which),
            the Ollama model manifest digest (/api/tags) — covers weights +
            template + parameters — and the GGUF weights blob digest parsed from
            the model's Modelfile FROM line (/api/show).

HONEST SCOPE (both): deployment-pinned re-execution. temp=0 greedy decode on the
same runtime + weights + params reproduces on the same deployment; verify() re-runs
it and compares output digests. Cross-machine bit-exactness is NOT claimed — that is
the exact-quire profile from Anomly's arithmetic work, not this. Ollama specifics:
the server picks the compute device (CPU / CUDA / Vulkan / Metal) and that choice
is part of the deployment; pin it with num_gpu when you need the receipt to survive
a GPU appearing or disappearing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request

LLAMACPP_PROFILE = "llamacpp-pinned-reexec-v0"
OLLAMA_PROFILE = "ollama-pinned-reexec-v0"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


def file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# --------------------------------------------------------------------------- llama.cpp

def run_llamacpp(binary: str, model: str, prompt: str,
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


class LlamaCppBackend:
    profile = LLAMACPP_PROFILE
    name = "llamacpp"

    def __init__(self, binary: str, model: str, threads: int = 4):
        self.binary, self.model, self.threads = binary, model, threads

    @property
    def model_name(self) -> str:
        return os.path.basename(self.model)

    def deployment(self) -> dict:
        return {"runtime_digest": file_digest(self.binary),
                "model_digest": file_digest(self.model),
                "model_name": self.model_name}

    def params(self, n_predict: int, seed: int) -> dict:
        return {"n_predict": n_predict, "seed": seed,
                "threads": self.threads, "temp": 0}

    def generate(self, prompt: str, params: dict) -> str:
        return run_llamacpp(self.binary, self.model, prompt,
                            params["n_predict"], params["seed"],
                            params.get("threads", self.threads))


# --------------------------------------------------------------------------- Ollama

class OllamaError(RuntimeError):
    pass


def _ollama_json(host: str, path: str, body: dict | None = None,
                 timeout: float = 600) -> dict:
    url = host.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise OllamaError(f"ollama {path} -> HTTP {e.code}: "
                          f"{e.read()[:300].decode(errors='replace')}") from None
    except urllib.error.URLError as e:
        raise OllamaError(f"ollama unreachable at {host}: {e.reason}") from None


def _normalize_tag(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def resolve_ollama_binary(host: str, explicit: str | None = None) -> str | None:
    """The ollama binary to hash as the runtime pin. Only meaningful when the
    server is local; a remote host's binary is not on this filesystem."""
    if explicit:
        return explicit
    env = os.environ.get("INVAR_OLLAMA_BIN")
    if env:
        return env
    local = any(h in host for h in ("127.0.0.1", "localhost", "[::1]"))
    return shutil.which("ollama") if local else None


class OllamaBackend:
    profile = OLLAMA_PROFILE
    name = "ollama"

    def __init__(self, model: str, host: str | None = None,
                 num_ctx: int = 2048, num_gpu: int | None = None,
                 binary: str | None = None):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST")
                     or DEFAULT_OLLAMA_HOST)
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.num_ctx, self.num_gpu = num_ctx, num_gpu
        self.binary = resolve_ollama_binary(self.host, binary)

    @property
    def model_name(self) -> str:
        return self.model

    # -- deployment identity -------------------------------------------------
    def deployment(self) -> dict:
        version = _ollama_json(self.host, "/api/version", timeout=10).get(
            "version", "unknown")
        if self.binary and os.path.exists(self.binary):
            runtime = file_digest(self.binary)
            pinned_by = "binary"
        else:
            runtime = "ollama-version:" + version
            pinned_by = "version"           # weaker; stated in the receipt
        tags = _ollama_json(self.host, "/api/tags", timeout=10).get("models", [])
        want = _normalize_tag(self.model)
        entry = next((m for m in tags
                      if _normalize_tag(m.get("name", "")) == want
                      or _normalize_tag(m.get("model", "")) == want), None)
        if entry is None:
            raise OllamaError(f"model {self.model!r} not present on {self.host} "
                              f"(ollama pull {self.model})")
        d = {"runtime_digest": runtime, "runtime_pinned_by": pinned_by,
             "runtime_version": version,
             "model_digest": "sha256:" + entry["digest"].removeprefix("sha256:"),
             "model_name": self.model}
        show = _ollama_json(self.host, "/api/show", {"model": self.model},
                            timeout=30)
        m = re.search(r"^FROM\s+.*sha256[-:]([0-9a-f]{64})\s*$",
                      show.get("modelfile", ""), re.M)
        if m:
            d["weights_digest"] = "sha256:" + m.group(1)
        return d

    def params(self, n_predict: int, seed: int) -> dict:
        p = {"n_predict": n_predict, "seed": seed, "temp": 0,
             "num_ctx": self.num_ctx}
        if self.num_gpu is not None:
            p["num_gpu"] = self.num_gpu
        return p

    def generate(self, prompt: str, params: dict) -> str:
        options = {"seed": params["seed"], "temperature": 0,
                   "num_predict": params["n_predict"],
                   "num_ctx": params.get("num_ctx", self.num_ctx)}
        if "num_gpu" in params:
            options["num_gpu"] = params["num_gpu"]
        r = _ollama_json(self.host, "/api/generate",
                         {"model": self.model, "prompt": prompt,
                          "stream": False, "options": options})
        if "response" not in r:
            raise OllamaError(f"ollama generate returned no response: "
                              f"{json.dumps(r)[:200]}")
        return r["response"]


# --------------------------------------------------------------------------- helpers

def looks_like_ollama_tag(model: str) -> bool:
    """`invar serve --model llama3.2` should just work: a model that is not a
    file on disk and has no gguf extension is an Ollama tag."""
    return not os.path.exists(model) and not model.lower().endswith(".gguf")


def make_backend(kind: str, model: str, *, binary: str | None = None,
                 host: str | None = None, threads: int = 4,
                 num_ctx: int = 2048, num_gpu: int | None = None):
    if kind == "auto":
        kind = "ollama" if looks_like_ollama_tag(model) else "llamacpp"
    if kind == "ollama":
        return OllamaBackend(model, host=host, num_ctx=num_ctx,
                             num_gpu=num_gpu, binary=binary)
    if kind == "llamacpp":
        return LlamaCppBackend(binary or os.environ.get("INVAR_LLAMA_BIN")
                               or shutil.which("llama-cli") or "llama-cli",
                               model, threads=threads)
    raise ValueError(f"unknown backend {kind!r}")


def backend_for_profile(profile: str):
    return {LLAMACPP_PROFILE: "llamacpp", OLLAMA_PROFILE: "ollama"}.get(profile)
