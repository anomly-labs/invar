# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.serve — OpenAI-compatible local endpoint where every completion
carries its worldline receipt. Stdlib only.

  POST /v1/chat/completions   {model?, messages:[{role,content}...], max_tokens?, stream?, tools?}
      tools present (or tool/tool_calls turns in messages) -> the Qwen2.5 chat template with
      the tools block is rendered HERE and run raw (params.chat="raw", certified), the
      model's <tool_call> JSON is returned as OpenAI `tool_calls` (finish_reason
      "tool_calls"); the receipt covers the exact rendered transcript.
      -> OpenAI-shaped response + "receipt": {certificate, chain, profile, manifest}
         stream=true -> SSE: one content chunk, one finish chunk (carrying the
         receipt), then [DONE]. The receipt covers the WHOLE output, so the
         output is produced first and streamed as a unit — no token-by-token
         trickle. Clients that insist on streaming (Open WebUI, aider, Continue)
         work; they just see the answer land at once.
  GET  /v1/models             {object:"list", data:[{id,...}]}   (Open WebUI needs it)
  GET  /v1/worldline          {entries, tip}
  GET  /v1/worldline/tail?n=N {entries:[last N full entries], tip}  (for SDKs
                              that drop unknown fields: LangChain, LlamaIndex)
  GET  /health                {ok, model, profile, backend}

Run:  invar serve --model <gguf path | ollama tag> [--backend auto|llamacpp|ollama]
                  [--binary <llama-cli>] [--ollama-host URL] [--num-gpu N]
                  [--port 8577] [--worldline path.jsonl]
Chat messages are flattened to a single pinned prompt (deterministic profile);
a proper chat template is a v1 item, stated honestly rather than faked. (On the
Ollama backend the model's own template wraps that prompt server-side, and the
template is covered by the pinned model digest.)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .attest import AttestationBinding
from .backends import LlamaCppBackend, make_backend
from .hwsign import make_signer
from .worldline import Worldline

_lock = threading.Lock()

_QWEN_SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def _render_chatml(msgs: list, tools: list | None) -> str:
    """Qwen2.5 chat template (tools variant), rendered here so the receipt covers the exact
    transcript the model saw. Mirrors tokenizer_config.json's Jinja: tools block in the system
    turn, assistant tool calls as <tool_call> JSON, tool results inside a user turn as
    <tool_response>, generation prompt appended. tojson == Jinja's htmlsafe_json_dumps (sorted keys,
    < > & ' escaped as \\uXXXX) — what HF apply_chat_template produced for the model's SFT data."""
    def tj(o):                     # Jinja's `tojson` (htmlsafe_json_dumps): sorted keys + the four HTML escapes
        return (json.dumps(o, sort_keys=True, ensure_ascii=False).replace("<", "\\u003c")
                .replace(">", "\\u003e").replace("&", "\\u0026").replace("'", "\\u0027"))
    out = []
    first_sys = msgs[0].get("content") if msgs and msgs[0].get("role") == "system" else None
    sys_text = _text_of(msgs[0]) if first_sys is not None else _QWEN_SYS
    if tools:
        out.append("<|im_start|>system\n" + sys_text +
                   "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                   "You are provided with function signatures within <tools></tools> XML tags:\n<tools>" +
                   "".join("\n" + tj(t) for t in tools) +
                   "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
                   "within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, "
                   "\"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")
    else:
        out.append("<|im_start|>system\n" + sys_text + "<|im_end|>\n")
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "system" and i == 0:
            continue
        if role == "tool":
            prev_tool = i > 0 and msgs[i - 1].get("role") == "tool"
            next_tool = i + 1 < len(msgs) and msgs[i + 1].get("role") == "tool"
            out.append(("" if prev_tool else "<|im_start|>user") + "\n<tool_response>\n" + _neutralise_control(_text_of(m)) +
                       "\n</tool_response>" + ("" if next_tool else "<|im_end|>\n"))
        elif role == "assistant" and m.get("tool_calls"):
            seg = "<|im_start|>assistant"
            if _text_of(m):
                seg += "\n" + _text_of(m)
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                seg += "\n<tool_call>\n{\"name\": \"" + str(fn.get("name")) + "\", \"arguments\": " + tj(args) + "}\n</tool_call>"
            out.append(seg + "<|im_end|>\n")
        else:
            body = _neutralise_control(_text_of(m)) if role == "user" else _text_of(m)
            out.append("<|im_start|>" + str(role) + "\n" + body + "<|im_end|>\n")
    out.append("<|im_start|>assistant\n")
    return "".join(out)


_ZWSP = "\u200b"


def _neutralise_control(text: str) -> str:
    """Untrusted content (user turns, tool results) must not be able to forge a turn: in raw mode
    the transcript is tokenised with special tokens honoured, so a tool result containing
    `<|im_start|>system` (or `<|start_header_id|>`, `[INST]`-style markers) would become control
    tokens. A zero-width space after the opening bracket keeps the text readable and makes it
    tokenise as plain text. The receipt covers the neutralised transcript — what the model saw."""
    if not text:
        return text
    return text.replace("<|", "<" + _ZWSP + "|").replace("[INST]", "[" + _ZWSP + "INST]").replace("[/INST]", "[" + _ZWSP + "/INST]")


def _hf_messages(msgs: list) -> list:
    """OpenAI messages -> the HF chat-template view (tool_calls arguments as dicts); user and
    tool content neutralised against control-token injection."""
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("role") in ("user", "tool") and isinstance(m.get("content"), str):
            m["content"] = _neutralise_control(m["content"])
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", tc))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except Exception:
                        pass
                tcs.append({**tc, "function": fn})
            m["tool_calls"] = tcs
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def _render_with_template(template: str, msgs: list, tools: list | None, tokens: dict) -> str | None:
    """Render with the MODEL'S OWN chat template (any family: Qwen, Llama 3.x, Mistral, ...) via
    Jinja2, the same engine HF used to build its SFT data. None when jinja2 is unavailable or the
    template fails, in which case the caller falls back to the Qwen hand-render."""
    try:
        import jinja2
        import datetime as _dt
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["strftime_now"] = lambda fmt: _dt.date.today().strftime(fmt)
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
        return env.from_string(template).render(messages=_hf_messages(msgs), tools=tools or None,
                                                add_generation_prompt=True, **tokens)
    except Exception:
        return None


def _gguf_template(path: str) -> tuple[str | None, dict]:
    """The GGUF's tokenizer.chat_template and its bos/eos token strings (header read only)."""
    try:
        from .spotcheck import GGUF
        kv = GGUF(path).kv
        toks = kv.get("tokenizer.ggml.tokens") or []
        def tk(key):
            i = int(kv.get(key, -1))
            return toks[i] if 0 <= i < len(toks) else ""
        return kv.get("tokenizer.chat_template"), {"bos_token": tk("tokenizer.ggml.bos_token_id"),
                                                    "eos_token": tk("tokenizer.ggml.eos_token_id")}
    except Exception:
        return None, {}


_TC_RX = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_MISTRAL_RX = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.S)
_LLAMA_TAG = "<|python_tag|>"


def _parse_tool_calls(text: str) -> tuple[str | None, list]:
    """Split the model's text into (content, OpenAI tool_calls). Unparseable blocks stay as content."""
    calls = []
    def _sub(m):
        try:
            obj = json.loads(m.group(1))
            name, args = obj["name"], obj.get("arguments", {})
        except Exception:
            return m.group(0)
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": str(name),
                                   "arguments": args if isinstance(args, str) else json.dumps(args)}})
        return ""
    content = _TC_RX.sub(_sub, text).strip()

    def _add(obj):
        name = obj.get("name")
        if not isinstance(name, str):                        # Llama-3.2-1B habit: {"function": "write", ...}
            name = obj.get("function") if isinstance(obj.get("function"), str) else None
        args = obj.get("arguments", obj.get("parameters", {}))
        if not name:
            return False
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": str(name), "arguments": args if isinstance(args, str) else json.dumps(args)}})
        return True
    if not calls and content:
        m = _MISTRAL_RX.search(content)                      # Mistral: [TOOL_CALLS] [{"name":..,"arguments":..}]
        if m:
            try:
                if all(_add(o) for o in json.loads(m.group(1)) if isinstance(o, dict)) and calls:
                    content = (content[:m.start()] + content[m.end():]).strip()
            except Exception:
                calls.clear()
        if not calls and _LLAMA_TAG in content:               # Llama 3.x: <|python_tag|>{"name":..,"parameters":..}
            body = content.split(_LLAMA_TAG, 1)[1].strip()
            try:
                objs = json.loads(body)
                objs = objs if isinstance(objs, list) else [objs]
                if all(_add(o) for o in objs if isinstance(o, dict)) and calls:
                    content = content.split(_LLAMA_TAG, 1)[0].strip()
            except Exception:
                calls.clear()
    if not calls and content:
        # lenient fallback: small models often answer with ONE bare or ```json-fenced
        # {"name":..., "arguments":{...}} object instead of the <tool_call> tags
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.S).strip()
        if (body.startswith("{") and body.endswith("}")) or (body.startswith("[") and body.endswith("]")):
            # repair: Python-style triple-quoted string values ("""...""") inside the JSON object
            body = re.sub(r'"""(.*?)"""', lambda m: json.dumps(m.group(1)), body, flags=re.S)
            try:
                obj = json.loads(body)
                objs = obj if isinstance(obj, list) else [obj]
                if objs and all(isinstance(o, dict) and (isinstance(o.get("name"), str) or isinstance(o.get("function"), str))
                                and ("arguments" in o or "parameters" in o) for o in objs):
                    for o in objs:
                        _add(o)
                    content = ""
            except Exception:
                pass
    return (content or None), calls


class _Server(ThreadingHTTPServer):
    # The OS default listen backlog (5) resets concurrent connects beyond it; a
    # receipted endpoint must queue honest burst traffic, not drop it.
    request_queue_size = 64


def _push_to_ledger(entry: dict) -> None:
    """Optional fleet push: LEDGER_URL + LEDGER_TOKEN (+ INVAR_DEVICE_ID) env.
    Best-effort by design — a Ledger outage must never fail local inference;
    the local worldline file remains the source of truth and can be re-pushed."""
    url = os.environ.get("LEDGER_URL")
    if not url:
        return
    try:
        import urllib.request
        body = json.dumps({
            "device_id": os.environ.get("INVAR_DEVICE_ID", "default"),
            "entries": [entry],
        }).encode()
        req = urllib.request.Request(
            url.rstrip("/") + "/v1/worldline/ingest", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ.get('LEDGER_TOKEN','')}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
    except Exception as e:
        print(f"[ledger push failed — kept locally] {e}", flush=True)


def _text_of(msg: dict) -> str:
    """OpenAI message content is a string or a list of parts; only text parts
    can be pinned into a prompt digest (images etc. are ignored, stated)."""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c
                         if isinstance(p, dict) and p.get("type") == "text"
                         and p.get("text"))
    return ""


def _approx_tokens(s: str) -> int:
    # usage is informational for clients that display it; not certified
    return max(1, len(s) // 4)


def make_handler(wl: Worldline, binary: str | None = None,
                 model: str | None = None, backend=None):
    """`backend` wins; (binary, model) is the original llama.cpp signature."""
    if backend is None:
        backend = LlamaCppBackend(binary, model)
    model_name = backend.model_name
    profile = backend.profile
    _tpl, _tpl_tokens = (_gguf_template(backend.model) if getattr(backend, "model", None)
                         and os.path.exists(str(getattr(backend, "model", ""))) else (None, {}))

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _sse_open(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self._sse_opened = True

        def _sse(self, events: list) -> None:
            if not getattr(self, "_sse_opened", False):
                self._sse_open()
            for ev in events:
                data = ev if isinstance(ev, str) else json.dumps(ev)
                self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()

        def log_message(self, *a):                     # quiet
            pass

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True, "model": model_name,
                                 "profile": profile, "backend": backend.name,
                                 "signer": getattr(wl.signer, "backend", None),
                                 "signer_key_id": getattr(wl.signer, "key_id", None),
                                 "host_attestation": wl.host_attestation,
                                 "genesis": wl.genesis})
            elif self.path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{
                    "id": model_name, "object": "model",
                    "created": 0, "owned_by": "invar"}]})
            elif self.path == "/v1/worldline":
                n = 0
                if os.path.exists(wl.path):
                    with open(wl.path) as f:
                        n = sum(1 for _ in f)
                self._json(200, {"entries": n, "tip": wl.tip})
            elif self.path.startswith("/v1/worldline/tail"):
                # last N entries, for clients whose SDK drops the receipt field
                # (LangChain, LlamaIndex): fetch right after your completion
                q = parse_qs(urlparse(self.path).query)
                try:
                    n = min(max(int(q.get("n", ["1"])[0]), 1), 100)
                except ValueError:
                    return self._json(400, {"error": "n must be an integer"})
                tail: list = []
                if os.path.exists(wl.path):
                    with open(wl.path) as f:
                        tail = [json.loads(x) for x in deque(f, maxlen=n)]
                self._json(200, {"entries": tail, "tip": wl.tip})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                return self._json(404, {"error": "not found"})
            try:
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 1_000_000:                     # 1 MB request cap
                    return self._json(413, {"error": "request too large"})
                req = json.loads(self.rfile.read(clen))
                msgs = req.get("messages") or []
                tools = req.get("tools") or None
                agentic = bool(tools) or any(m.get("role") == "tool" or m.get("tool_calls") for m in msgs)
                if agentic:
                    # tool use needs the real chat template (tools block, tool_call/tool_response turns);
                    # rendered here and run raw so the receipt digests the exact transcript.
                    prompt = (_render_with_template(_tpl, msgs, tools, _tpl_tokens) if _tpl else None) \
                        or _render_chatml(msgs, tools)
                    chat_mode = "raw"
                else:
                    prompt = "\n".join(t for t in (_text_of(m) for m in msgs) if t)
                    chat_mode = None
                if not prompt:
                    return self._json(400, {"error": "empty prompt"})
                n = min(max(int(req.get("max_tokens")
                                or req.get("max_completion_tokens") or 128),
                            1), 4096)
                if len(prompt) > 32_768:
                    return self._json(400, {"error": "prompt too long (32k max)"})
                if req.get("stream"):
                    # open the stream NOW and heartbeat while the pinned run executes: a slow
                    # (CPU, exact-profile) run must not trip the client's idle timeout; the
                    # receipt still covers the whole output, delivered as one chunk at the end.
                    self._sse_open()
                    box: dict = {}

                    def _run():
                        try:
                            with _lock:                     # one pinned run at a time
                                box["r"] = wl.infer(backend, prompt, n_predict=n, chat=chat_mode)
                        except Exception as ex:             # noqa: BLE001 — surfaced on the stream below
                            box["e"] = ex
                    th = threading.Thread(target=_run, daemon=True)
                    th.start()
                    while th.is_alive():
                        th.join(10)
                        if th.is_alive():
                            try:
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                            except OSError:                 # client went away; the run finishes and is receipted anyway
                                th.join()
                                return
                    if "e" in box:
                        self._sse([{"error": str(box["e"])}, "[DONE]"])
                        return
                    text, entry = box["r"]
                else:
                    with _lock:                             # one pinned run at a time
                        text, entry = wl.infer(backend, prompt, n_predict=n, chat=chat_mode)
                content, tool_calls = _parse_tool_calls(text) if agentic else (text, [])
                finish = "tool_calls" if tool_calls else "stop"
                _push_to_ledger(entry)                  # best-effort, never blocks the answer
                receipt = {"certificate": entry["certificate"],
                           "chain": entry["chain"],
                           "profile": profile,
                           "manifest": entry["manifest"]}
                if entry.get("signature"):
                    # OpenPCC-shaped evidence piece: {Type, Data, Signature}. Data is the
                    # entry JSON (certified manifest + certificate + chain) and Signature is
                    # the device key's signature over the chain digest; a client verifies it
                    # with go/crverify VerifyExecutionReceipt against the node's attestation.
                    receipt["openpcc"] = {
                        "type": "ExecutionReceipt",
                        "data": json.dumps({k: entry[k] for k in ("manifest", "certificate", "chain")},
                                           separators=(",", ":"), sort_keys=True),
                        "signature": entry["signature"]}
                rid = "chatcmpl-" + uuid.uuid4().hex[:24]
                now = int(time.time())
                usage = {"prompt_tokens": _approx_tokens(prompt),
                         "completion_tokens": _approx_tokens(text),
                         "total_tokens": _approx_tokens(prompt) + _approx_tokens(text)}
                if req.get("stream"):
                    head = {"id": rid, "object": "chat.completion.chunk",
                            "created": now, "model": model_name}
                    delta = {"role": "assistant", "content": content}
                    if tool_calls:
                        delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(tool_calls)]
                    self._sse([
                        {**head, "choices": [{"index": 0, "finish_reason": None, "delta": delta}]},
                        {**head, "choices": [{"index": 0, "finish_reason": finish,
                                              "delta": {}}],
                         "usage": usage, "receipt": receipt},
                        "[DONE]",
                    ])
                    return
                message = {"role": "assistant", "content": content}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                self._json(200, {
                    "id": rid, "object": "chat.completion", "created": now,
                    "model": model_name,
                    "choices": [{"index": 0, "finish_reason": finish,
                                 "message": message}],
                    "usage": usage,
                    "receipt": receipt,
                })
            except Exception as e:                      # surface, don't hide
                if getattr(self, "_sse_opened", False):
                    try:
                        self._sse([{"error": str(e)}, "[DONE]"])
                    except OSError:
                        pass
                else:
                    self._json(500, {"error": str(e)})
    return Handler


def add_backend_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--backend", choices=["auto", "llamacpp", "ollama", "openai", "witness"],
                    default="auto",
                    help="auto = gguf path -> llamacpp, anything else -> ollama")
    ap.add_argument("--binary", default=None,
                    help="llama.cpp binary (INVAR_LLAMA_BIN or PATH), or the "
                         "ollama binary to hash as the runtime pin (INVAR_OLLAMA_BIN)")
    ap.add_argument("--ollama-host", default=None,
                    help="Ollama server (default OLLAMA_HOST or http://127.0.0.1:11434)")
    ap.add_argument("--upstream-url", default=None,
                    help="openai/witness backends: base URL of an OpenAI-compatible endpoint "
                         "(vLLM/SGLang/TGI you operate -> 'openai' = pinned, replayable; a closed "
                         "provider -> 'witness' = provenance record only). Key: INVAR_UPSTREAM_API_KEY")
    ap.add_argument("--weights-dir", default=None,
                    help="openai backend: local checkpoint directory of the served model; its "
                         "digest becomes the receipt's weights_digest (otherwise an identifier only)")
    ap.add_argument("--image-digest", default=None,
                    help="openai backend: container image digest of the engine (sha256:...) to pin "
                         "the runtime by image rather than by version string")
    ap.add_argument("--num-ctx", type=int, default=2048,
                    help="Ollama context size to pin (default 2048)")
    ap.add_argument("--num-gpu", type=int, default=None,
                    help="Ollama layers on GPU to pin (0 = CPU only); unset = "
                         "server decides, and that decision is part of the deployment")
    ap.add_argument("--threads", type=int, default=4,
                    help="llama.cpp thread count to pin (default 4)")
    ap.add_argument("--device", default=None,
                    help="llama.cpp compute device to pin: none = CPU only, or e.g. "
                         "CUDA0 (see llama-cli --list-devices); unset = binary default")
    ap.add_argument("--ngl", type=int, default=None,
                    help="llama.cpp layers to offload to --device (pinned in the receipt)")


def backend_from_args(a, model: str):
    return make_backend(a.backend, model, binary=a.binary, host=a.ollama_host,
                        threads=a.threads, num_ctx=a.num_ctx, num_gpu=a.num_gpu,
                        device=a.device, n_gpu_layers=a.ngl,
                        upstream_url=getattr(a, "upstream_url", None),
                        weights_dir=getattr(a, "weights_dir", None),
                        image_digest=getattr(a, "image_digest", None))


def main():
    ap = argparse.ArgumentParser(description="receipted local inference server")
    ap.add_argument("--model", required=True,
                    help="gguf path (llama.cpp) or model tag (Ollama, e.g. llama3.2)")
    add_backend_args(ap)
    ap.add_argument("--port", type=int, default=8577)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default loopback; expose deliberately)")
    ap.add_argument("--worldline", default="worldline.jsonl")
    ap.add_argument("--signer", default=os.environ.get("INVAR_SIGNER"),
                    help="sign every entry: software | tpm2 | tpm2:sha256:0,7 "
                         "(PCR-policy-bound TPM key). Default: unsigned.")
    ap.add_argument("--state-dir", default=os.environ.get("INVAR_STATE",
                    os.path.expanduser("~/.invar")),
                    help="where signing keys / TPM contexts live")
    ap.add_argument("--spot-check", action="store_true",
                    help="exact profile only: keep a per-request logits dump (content-addressed, "
                         "beside the worldline) and certify its digest so a client can "
                         "re-execute challenged lm_head rows (docs/SPOT-CHECK.md)")
    ap.add_argument("--spot-check-units", action="store_true",
                    help="with --spot-check: also capture every layer's FFN/attn-out matmul "
                         "inputs+outputs so verify --units can re-execute them")
    ap.add_argument("--spot-check-keep", type=int, default=1000,
                    help="retain this many newest spot-check dumps (0 = unlimited)")
    ap.add_argument("--attest", default=os.environ.get("INVAR_ATTEST"),
                    help="attestation binding JSON (invar attest bind ...): genesis and "
                         "every receipt commit to the platform evidence")
    a = ap.parse_args()
    backend = backend_from_args(a, a.model)
    if a.spot_check:
        if getattr(backend, "profile", "") != "llamacpp-bposit8-quire-v0":
            ap.error("--spot-check needs the exact profile (a b-posit8 GGUF on llama-cpp-et)")
        backend.dumps_dir = a.worldline + ".dumps"
        backend.dumps_keep = a.spot_check_keep
        backend.dump_units = a.spot_check_units
    dep = backend.deployment()          # fail fast: unreachable server / missing model
    os.makedirs(a.state_dir, mode=0o700, exist_ok=True)
    signer = make_signer(a.signer, a.state_dir)
    binding = AttestationBinding.load(a.attest) if a.attest else None
    wl = Worldline(a.worldline, signer=signer, binding=binding)
    srv = _Server((a.host, a.port), make_handler(wl, backend=backend))
    pins = ", ".join(f"{k}={dep[k][:23]}…" if len(dep[k]) > 30 else f"{k}={dep[k]}"
                     for k in ("runtime_digest", "model_digest", "weights_digest")
                     if k in dep)
    print(f"receipted endpoint on {a.host}:{a.port}  backend={backend.name} "
          f"model={backend.model_name} profile={backend.profile} "
          f"worldline={a.worldline}\n  pins: {pins}"
          + (f"\n  signer: {signer.backend} key {signer.key_id}" if signer else "")
          + (f"\n  attestation: {binding.kind} genesis {wl.genesis[:30]}…"
             if binding else ""))
    srv.serve_forever()


if __name__ == "__main__":
    main()
