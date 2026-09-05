Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.

# Using INVAR from the tools you already run

INVAR is an OpenAI-compatible endpoint. Anything that can talk to
`https://api.openai.com/v1` can talk to `http://127.0.0.1:8577/v1` instead, and
every answer it gets back carries a receipt. This page is the exact config for
the common clients. Each was checked against a running INVAR; where a client
drops the receipt field, it says so and shows where to get it instead.

Start the endpoint first (either backend):

```
invar serve --model llama3.2               # Ollama tag  (needs `ollama pull llama3.2`)
invar serve --model ~/models/your.gguf     # llama.cpp gguf
```

The endpoint listens on loopback (`127.0.0.1:8577`) by default. The API key is
not checked; clients that insist on one can send anything (`invar` below).

---

## Ollama (as the engine behind INVAR)

You keep Ollama exactly as it is. INVAR sits in front of it, pins what ran, and
verifies later by asking Ollama to run it again.

```
ollama pull llama3.2
invar serve --model llama3.2            # auto-detects: not a file -> Ollama backend
invar verify worldline.jsonl            # re-executes every entry against Ollama
```

What the receipt pins on the Ollama backend:

| Field | Source | Meaning |
|---|---|---|
| `runtime_digest` | sha256 of the `ollama` binary | **when the binary is on PATH** (or `INVAR_OLLAMA_BIN`); otherwise `ollama-version:<v>` from `/api/version`, and `runtime_pinned_by` says `"version"` so a reader knows it is the weaker pin |
| `model_digest` | `/api/tags` digest | the model manifest: weights + template + parameters together |
| `weights_digest` | `FROM` line of the Modelfile (`/api/show`) | the GGUF blob itself |
| `params` | request | `seed`, `temp=0`, `n_predict`, `num_ctx`, and `num_gpu` if you pinned it |

Options that matter:

- `--ollama-host URL` (or `OLLAMA_HOST`): a remote Ollama works, but its binary
  is not on your disk, so the runtime pin falls back to version-only.
- `--num-gpu N`: Ollama decides at load time how many layers go to a GPU, and
  that decision is part of the deployment. If a GPU can appear or disappear on
  the box (laptops, shared servers), pin it: `--num-gpu 0` for CPU-only receipts.
- `--num-ctx N`: pinned into the receipt and replayed verbatim (default 2048).
- Run Ollama with `OLLAMA_NUM_PARALLEL=1` for receipted deployments. Batching
  unrelated requests together can change floating-point reduction order on some
  backends; INVAR serialises its own requests, but other clients of the same
  Ollama may not.

Verification needs only the server: `invar verify worldline.jsonl` reads the
model tag out of each receipt. Add `--ollama-host` for a non-default server and
`--binary /path/to/ollama` (or `INVAR_OLLAMA_BIN`) if the binary is not on PATH.

Docker: the shipped image includes llama.cpp, not Ollama. To receipt an Ollama
that runs on the host, run the container on the host network:
`docker run --network host anomly/invar invar serve --model llama3.2 --worldline /data/worldline.jsonl`.

---

## Open WebUI

Open WebUI lists models from `GET /v1/models` and streams by default; INVAR
serves both.

Simplest (Open WebUI in Docker on the same machine):

```
docker run -d --network host \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:8577/v1 \
  -e OPENAI_API_KEY=invar \
  -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
```

Or through the UI: **Admin Panel → Settings → Connections → OpenAI API → +**,
URL `http://127.0.0.1:8577/v1`, key `invar`. Without `--network host` the
container cannot reach a loopback-bound INVAR; use
`--add-host=host.docker.internal:host-gateway`, point the URL at
`http://host.docker.internal:8577/v1`, and start INVAR with `--host 0.0.0.0`
behind a firewall.

Streaming note: the receipt covers the whole answer, so INVAR produces the
answer first and streams it as one chunk. You see the reply land at once rather
than token by token. The receipt travels on the final SSE chunk.

Where the receipt is: Open WebUI does not display extra response fields. Every
answer is in `worldline.jsonl` beside the server, and `invar verify` checks it.

---

## aider

```
aider --openai-api-base http://127.0.0.1:8577/v1 \
      --openai-api-key invar \
      --model openai/llama3.2 \
      --no-show-model-warnings
```

The `openai/` prefix tells aider's LiteLLM layer to use the OpenAI protocol
against your base URL; the part after it must match the model INVAR is serving
(`invar serve --model llama3.2` → `openai/llama3.2`). Small local models cope
better with `--edit-format whole`.

Every edit aider makes now has a receipt in `worldline.jsonl` proving which
model, weights, and prompt produced it. That is the answer to "which model wrote
this commit?".

---

## Continue (VS Code / JetBrains)

`~/.continue/config.yaml`:

```yaml
models:
  - name: INVAR llama3.2
    provider: openai
    model: llama3.2
    apiBase: http://127.0.0.1:8577/v1
    apiKey: invar
    roles: [chat, edit]
```

Older `config.json` installs use the same fields (`"provider": "openai"`,
`"apiBase"`, `"apiKey"`, `"model"`).

---

## OpenAI Python SDK (receipt in-process)

The SDK keeps unknown top-level fields, so the receipt is one attribute away.
Checked against `openai==2.43`:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8577/v1", api_key="invar")

r = c.chat.completions.create(model="llama3.2",
        messages=[{"role": "user", "content": "The capital of France is"}])
print(r.choices[0].message.content)
receipt = r.model_extra["receipt"]           # certificate, chain, profile, manifest
print(receipt["certificate"], receipt["profile"])

# streaming: the receipt rides on the final chunk
last = None
for chunk in c.chat.completions.create(model="llama3.2", stream=True,
        messages=[{"role": "user", "content": "The capital of Italy is"}]):
    last = chunk
print(last.model_extra["receipt"]["chain"])
```

Node (`openai` npm) behaves the same: the parsed response object carries a
`receipt` property.

---

## LangChain / LlamaIndex

Both connect through their OpenAI-compatible wrappers:

```python
# LangChain
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="llama3.2", base_url="http://127.0.0.1:8577/v1", api_key="invar")

# LlamaIndex
from llama_index.llms.openai_like import OpenAILike
llm = OpenAILike(model="llama3.2", api_base="http://127.0.0.1:8577/v1",
                 api_key="invar", is_chat_model=True)
```

These wrappers drop response fields they do not know, so the receipt is not on
the returned message. Two ways to get it:

- `GET http://127.0.0.1:8577/v1/worldline/tail?n=1` returns the most recent
  entries (certificate, chain, manifest, and the evidence texts). Call it right
  after the completion; INVAR serialises requests, so the last entry is yours.
- Or read `worldline.jsonl` directly; it is append-only JSON lines.

---

## curl

```
curl -s localhost:8577/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":32}' \
  | python3 -m json.tool
```

---

## llama.cpp on a GPU (CUDA)

```
invar serve --model model-bposit8.gguf --binary llama-cli --device CUDA0 --ngl 99 --spot-check --spot-check-units
invar verify worldline.jsonl --binary llama-cli --model model-bposit8.gguf --device CUDA0 --ngl 99 --spot-check --units
```

The compute device and the number of offloaded layers are part of the deployment pin
(`device`, `n_gpu_layers` in the manifest). Under the exact profile the whole graph is
bit-identical between the CPU and CUDA (docs/DETERMINISTIC-GRAPH.md), so a receipt minted
on CUDA0 re-executes on the CPU with `invar verify --cross-deployment`; without the flag
the pin is enforced, and float-profile receipts never cross. Use `--device none` to pin a
CPU-only run on a machine that has a GPU; with a CUDA build and no `--device`, llama.cpp
may offload batched prefill ops on its own, which is a deployment property you did not
pin. New receipts also certify `flash_attn: off` (the exact attention path).

## What every client gets, and what none of them get

Every answer through any of these clients is pinned to the runtime, weights,
prompt, and parameters that produced it, hash-chained into the worldline, and
re-executable with `invar verify`. What no client gets: a judgement that the
answer is good, or a receipt that survives moving the workload to different
hardware. The default profiles pin a deployment; cross-machine bit-exactness is
a separate exact-arithmetic profile. Full boundary: [THREATMODEL.md](THREATMODEL.md).
