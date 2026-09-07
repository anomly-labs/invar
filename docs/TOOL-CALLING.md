<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
# Tool calling through `invar serve` (agents, every turn receipted)

`POST /v1/chat/completions` accepts the OpenAI `tools` field and returns `tool_calls`. A coding
agent (Pi, OpenHands, aider's agent mode, your own loop) therefore runs unchanged against INVAR,
and **every turn — including the ones that decide to call a tool — is a receipt** that a third
party can re-execute from the weights and the exact transcript.

## What happens on a tools request

1. **The transcript is rendered by INVAR**, not by llama.cpp, with the **model's own chat
   template** read from the GGUF header and rendered through Jinja2 — the same engine HF used to
   build the model's SFT data — with `tools` passed in. Any family whose template handles `tools`
   (Qwen2.5, Llama 3.x, Mistral, ...) therefore gets its native tools block, tool-call turns and
   tool-result turns. If Jinja2 is unavailable or the template fails, a hand render of the Qwen2.5
   tools template is used; that hand render is tested byte-identical to the Jinja output,
   including Jinja's HTML-safe `tojson` escaping (`tests/test_tools.py`). The receipt's
   `prompt_text` is this exact transcript.
2. **The run is raw.** llama-cli's single-turn mode always pushes `-p` through the model's chat
   template as a user turn (measured: a 13-token ChatML transcript arrives as 42 tokens, nested).
   Raw mode substitutes an identity Jinja template (`{{ messages[0]['content'] }}`), so the
   transcript is tokenised verbatim, special tokens parsed. This is certified in the receipt as
   `computation.params.chat = "raw"`.
3. **The verifier honours the mode.** `invar verify` re-executes with the receipt's params, so a
   raw receipt re-runs raw; the tokenisation check re-tokenises the transcript raw too (the
   identity template drops exactly one trailing newline; measured, mirrored).
4. **The model's tool call becomes `tool_calls`.** Recognised shapes: Qwen `<tool_call>{json}
   </tool_call>` blocks; Mistral `[TOOL_CALLS] [{...}]`; Llama 3.x `<|python_tag|>{"name":..,
   "parameters":..}`; and, for small models, one bare or ```json-fenced object or list (Python
   triple-quoted string values are repaired). Text outside the call is the `content`;
   `finish_reason` is `"tool_calls"`. An unparseable block stays visible as content rather than
   being silently dropped.
5. **Streaming** delivers one content/tool_calls chunk when the run completes, with `:
   keepalive` comments every 10 s meanwhile (slow CPU exact-profile turns must not trip client
   idle timeouts — an agent that retries burns receipted runs).

Requests without `tools` (and without `tool`/`tool_calls` turns) are unchanged: the flattened
prompt is run through llama-cli's template as before, so existing worldlines keep verifying.

## Verified example (2026-09-07, CPU exact profile, 1.5B model)

Pi 0.85.1, `--tools write,bash`, task "create hello.py printing hello from invar, then run it":
turn 1 `write(path, content)`, turn 2 `bash("python3 hello.py")` → `hello from invar`, turn 3
final answer. Three receipts, zero retries, `invar verify --reexec` re-executes each turn.

## Verified example 2 (2026-09-07, second model family)

Llama-3.2-1B-Instruct (b-posit8 exact GGUF, CPU): INVAR rendered the GGUF's own Llama 3.2
template — `Environment: ipython` system turn, the functions block, "Respond in the format
{"name": ..., "parameters": ...}" — ran it raw, the model answered
`{"function": "write", "parameters": {"path": "hi.txt", "content": "hello"}}`, parsed as a
`write` tool call, receipt certified `chat=raw`, `invar verify --reexec` ACCEPT. Same code path
as Qwen; nothing family-specific was configured.

## Limits (stated, not hidden)

- Rendering follows the GGUF's own template, so a model whose template ignores `tools` (many
  base/older instruct templates) gets no tools block and will not call tools; the receipt still
  covers exactly what it saw. Tool-call *parsing* covers Qwen, Mistral and Llama 3.x shapes; a
  family with a different output convention returns its call as plain content.
- `tool_choice`, `parallel_tool_calls`, `response_format` are accepted and ignored.
- A receipt certifies *what the model was asked and what it answered*, including the tool call
  it requested. What the agent then did with that call (the file write, the shell command) is
  outside the model receipt — record it in your agent's own log or a computation receipt of the
  tool step.
