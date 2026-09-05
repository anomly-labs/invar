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
LLAMACPP_EXACT_PROFILE = "llamacpp-bposit8-quire-v0"
OLLAMA_PROFILE = "ollama-pinned-reexec-v0"
GGUF_FTYPE_BPOSIT8 = 42          # LLAMA_FTYPE_MOSTLY_BPOSIT8 in llama-cpp-et
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


def file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# --------------------------------------------------------------------------- GGUF sniffing

_GGUF_TYPES = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
               6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}


def gguf_file_type(path: str) -> int | None:
    """Read `general.file_type` from a GGUF v2/v3 header (stdlib, streams only the KV
    section). Returns None if the file is not GGUF or the key is absent."""
    import struct
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            ver, = struct.unpack("<I", f.read(4))
            if ver < 2:
                return None
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

            def rd_str():
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")

            def rd_val(t):
                if t == 8:
                    return rd_str()
                if t == 9:
                    et, = struct.unpack("<I", f.read(4))
                    cnt, = struct.unpack("<Q", f.read(8))
                    return [rd_val(et) for _ in range(cnt)]
                fmt, sz = _GGUF_TYPES[t]
                return struct.unpack("<" + fmt, f.read(sz))[0]

            for _ in range(n_kv):
                key = rd_str()
                t, = struct.unpack("<I", f.read(4))
                val = rd_val(t)
                if key == "general.file_type":
                    return int(val)
    except (OSError, struct.error, KeyError, UnicodeDecodeError):
        return None
    return None


# --------------------------------------------------------------------------- llama.cpp

def run_llamacpp(binary: str, model: str, prompt: str,
                 n_predict: int = 128, seed: int = 1, threads: int = 4,
                 logits_out: str | None = None, logits_layers: bool = False,
                 logits_matmuls: bool = False) -> str:
    """Deterministically-pinned llama.cpp run; returns ONLY the generated text.

    Uses single-turn simple-io mode (-st --simple-io) — this llama.cpp line ships
    an interactive chat UI that blocks on stdin without -st. The UI echoes the
    prompt as a "> ..." line and appends a "[ Prompt: ... t/s ]" stats line whose
    numbers vary run-to-run, so the receipt digests the EXTRACTED generation only.
    """
    cmd = [binary, "-m", model, "-p", prompt, "-n", str(n_predict),
           "--temp", "0", "--seed", str(seed), "-t", str(threads),
           "-st", "--simple-io"]
    env = dict(os.environ)
    env.pop("INVAR_LOGITS_OUT", None)
    if logits_out:
        env["INVAR_LOGITS_OUT"] = logits_out       # llama-cpp-et CSC hook (see spotcheck.py)
        if logits_layers:
            env["INVAR_LOGITS_LAYERS"] = "1"       # also capture per-layer l_out rows
        if logits_matmuls:
            env["INVAR_LOGITS_MATMULS"] = "1"      # inputs/outputs of every FFN/attn-out matmul
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                         stdin=subprocess.DEVNULL, env=env)
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
    """profile is chosen from the MODEL: a GGUF whose file_type is b-posit8 (42) runs
    every matmul through the exact 256-bit quire in llama-cpp-et, so its receipts carry
    the `llamacpp-bposit8-quire-v0` profile — arithmetic that is order-independent by
    construction, hence re-executable across implementations, not just on the pinned
    deployment. Any other GGUF gets the deployment-pinned profile. The profile is
    certified in the manifest, so a verifier always knows which guarantee it holds."""
    name = "llamacpp"

    def __init__(self, binary: str, model: str, threads: int = 4,
                 profile: str | None = None, dumps_dir: str | None = None,
                 dumps_keep: int = 1000):
        self.binary, self.model, self.threads = binary, model, threads
        self.dumps_dir = dumps_dir            # when set (exact profile): capture CSC dumps
        self.dumps_keep = dumps_keep          # retention: oldest dumps beyond this are removed
        self.dump_units = False               # also capture per-matmul units (INVAR_LOGITS_MATMULS)
        self.last_dump: str | None = None
        if profile is None:
            ft = gguf_file_type(model) if model and os.path.exists(model) else None
            profile = LLAMACPP_EXACT_PROFILE if ft == GGUF_FTYPE_BPOSIT8 else LLAMACPP_PROFILE
        self.profile = profile
        self.file_type = gguf_file_type(model) if model and os.path.exists(model) else None

    @property
    def model_name(self) -> str:
        return os.path.basename(self.model)

    def deployment(self) -> dict:
        d = {"runtime_digest": file_digest(self.binary),
             "model_digest": file_digest(self.model),
             "model_name": self.model_name}
        if self.file_type is not None:
            d["gguf_file_type"] = self.file_type
        return d

    def params(self, n_predict: int, seed: int) -> dict:
        return {"n_predict": n_predict, "seed": seed,
                "threads": self.threads, "temp": 0}

    def generate(self, prompt: str, params: dict) -> str:
        self.last_dump = None
        dump = None
        if self.dumps_dir and self.profile == LLAMACPP_EXACT_PROFILE:
            os.makedirs(self.dumps_dir, exist_ok=True)
            import tempfile
            fd, dump = tempfile.mkstemp(prefix="dump-", suffix=".jsonl", dir=self.dumps_dir)
            os.close(fd)
            os.unlink(dump)                    # the hook appends; start from nothing
        text = run_llamacpp(self.binary, self.model, prompt,
                            params["n_predict"], params["seed"],
                            params.get("threads", self.threads), logits_out=dump,
                            logits_matmuls=self.dump_units)
        if dump and os.path.exists(dump):
            self.last_dump = dump
        return text

    def spot_check_field(self) -> dict | None:
        """Certified into the manifest after generate(): the dump's digest and size.
        The verifier picks the challenge later (commit-before-challenge)."""
        if not self.last_dump:
            return None
        from .spotcheck import dump_digest, read_dump
        d = dump_digest(self.last_dump)
        n = len(read_dump(self.last_dump))
        final = os.path.join(self.dumps_dir, d.split(":", 1)[1] + ".jsonl")
        os.replace(self.last_dump, final)      # content-addressed evidence file
        self.last_dump = final
        self._prune_dumps()
        what = "last-row result_norm+result_output per eval"
        if self.dump_units:
            what += " + per-layer matmul unit inputs/outputs"
        return {"dump_digest": d, "n_evals": n, "what": what, "units": bool(self.dump_units)}

    def _prune_dumps(self) -> None:
        """Keep the newest `dumps_keep` dumps. A pruned dump makes its receipt's spot-check
        'dump missing' (structure + re-execution still verify); the digest stays certified."""
        if not self.dumps_keep or self.dumps_keep < 0:
            return
        try:
            files = [os.path.join(self.dumps_dir, f) for f in os.listdir(self.dumps_dir)
                     if f.endswith(".jsonl") and not f.startswith("dump-")]
            files.sort(key=os.path.getmtime)
            for old in files[:-self.dumps_keep] if len(files) > self.dumps_keep else []:
                os.unlink(old)
        except OSError:
            pass


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
    return {LLAMACPP_PROFILE: "llamacpp", LLAMACPP_EXACT_PROFILE: "llamacpp",
            OLLAMA_PROFILE: "ollama"}.get(profile)
