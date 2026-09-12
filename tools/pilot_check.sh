#!/bin/sh
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
# Licensed under the Apache License, Version 2.0 (same as the invar repository).
#
# pilot_check.sh -- the first ten minutes of a verified-inference pilot, as one command.
#
#   tools/pilot_check.sh --upstream http://host:port --model NAME        # attested tier: your server
#   tools/pilot_check.sh --gguf model.gguf [--binary llama-cli] [--device MTL0 --ngl 99]   # exact tier: INVAR serves it
#   add --witness for a CLOSED provider behind an API key (provenance record; nothing can re-execute it)
#
# What it does, in order, printing a one-line result for each step:
#   1. triage      run tools/determinism_triage.py against the server that will answer (before picture)
#   2. serve       start `invar serve` (in front of your server, or on the exact tier) on a spare port
#   3. receipt     ask one question through it; show the receipt's profile and digests
#   4. verify      `invar verify` the worldline by re-execution (ACCEPT / INDETERMINATE / REJECT)
#   5. tamper      alter the readable output copy in a COPY of the log and verify again (must REJECT)
#   6. summary     what was proven, what was not, and what to do next
# Nothing on your server is changed. Everything lands in ./pilot-check-<timestamp>/.
set -eu
WITNESS=""; UP=""; MODEL=""; GGUF=""; BINARY="${INVAR_LLAMA_BIN:-llama-cli}"; PORT="${PILOT_PORT:-8577}"; PROMPT="Explain in three sentences why floating-point summation depends on operand order."
DEVICE="${INVAR_DEVICE:-}"; NGL="${INVAR_NGL:-}"   # exact tier: llama.cpp device / offloaded layers (e.g. --device MTL0 --ngl 99 on an Apple GPU build)
while [ $# -gt 0 ]; do case "$1" in
  --upstream) UP="$2"; shift 2;; --model) MODEL="$2"; shift 2;; --gguf) GGUF="$2"; shift 2;;
  --binary) BINARY="$2"; shift 2;; --port) PORT="$2"; shift 2;; --prompt) PROMPT="$2"; shift 2;;
  --device) DEVICE="$2"; shift 2;; --ngl) NGL="$2"; shift 2;;
  --witness) WITNESS="--backend witness"; shift;;
  *) echo "unknown arg $1"; exit 2;; esac; done
[ -n "$UP" ] || [ -n "$GGUF" ] || { echo "need --upstream URL --model NAME, or --gguf FILE"; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="pilot-check-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"; cd "$OUT"
INVAR="${INVAR_BIN:-invar}"; command -v "$INVAR" >/dev/null || { echo "invar not on PATH (curl -fsSL https://www.anomly.com/get/invar.sh | sh)"; exit 2; }
say() { printf '%s\n' "$*" | tee -a summary.txt; }
say "pilot check $(date -u +%FT%TZ) -- $( [ -n "$UP" ] && echo "$( [ -n "$WITNESS" ] && echo witness || echo attested) tier, upstream $UP model $MODEL" || echo "exact tier, gguf $GGUF" )"

# 1. triage (before picture) -- against the upstream for the attested tier
if [ -n "$UP" ]; then
  python3 "$HERE/determinism_triage.py" --url "$UP" --model "$MODEL" --n 6 --max-tokens 48 --concurrent --out triage-upstream.json > triage-upstream.txt 2>&1 || true
  say "1. triage (your server): $(grep '^summary:' triage-upstream.txt | sed 's/^summary: //')"
else
  say "1. triage: skipped (exact tier serves the model itself; see step 4)"
fi

# 2. serve (refuse a port something else already answers on: the receipt would come from that server)
if curl -s -m 2 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
  say "2. serve: port $PORT already answers; pass --port N (or PILOT_PORT=N) for a free one"; exit 1; fi
export INVAR_STATE="$PWD/.invar"
if [ -n "$UP" ]; then "$INVAR" serve $WITNESS --upstream-url "$UP" --model "$MODEL" --port "$PORT" > serve.log 2>&1 &
else DEVARGS=""; [ -n "$DEVICE" ] && DEVARGS="--device $DEVICE"; [ -n "$NGL" ] && DEVARGS="$DEVARGS --ngl $NGL"
  "$INVAR" serve --model "$GGUF" --binary "$BINARY" $DEVARGS --port "$PORT" > serve.log 2>&1 & fi
SPID=$!; trap 'kill $SPID 2>/dev/null || true' EXIT
i=0; while [ $i -lt 60 ]; do curl -s -m 2 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 1; i=$((i+1)); done
curl -s -m 2 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 || { say "2. serve: FAILED to start (see $OUT/serve.log)"; exit 1; }
say "2. serve: invar serve up on :$PORT $(grep -i 'self-test' serve.log | head -1 | sed 's/^/-- /')"

# 3. receipt
t0=$(date +%s)
# (built without nested double quotes: bash 3.2 on macOS brace-expands the inner python literal otherwise)
BODY=$(PILOT_PROMPT="$PROMPT" python3 -c 'import json,os; print(json.dumps({"messages":[{"role":"user","content":os.environ["PILOT_PROMPT"]}],"max_tokens":48,"temperature":0}))')
curl -s -m 600 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" > resp.json
python3 - <<'PY' | tee -a summary.txt
import json; d=json.load(open("resp.json")); r=d.get("receipt") or {}; m=r.get("manifest") or {}
print("3. receipt: profile", m.get("profile"), "| model", (m.get("computation") or {}).get("model_name"),
      "| output", (m.get("outputs") or {}).get("text","")[:23], "| self-test", (m.get("computation") or {}).get("upstream_self_test","n/a"))
print("   answer:", (d["choices"][0]["message"]["content"] or "")[:100].replace("\n"," "))
PY
say "   first receipted answer in $(( $(date +%s) - t0 )) s"

# 4. verify (re-execution)
if [ -n "$UP" ]; then "$INVAR" verify worldline.jsonl --upstream-url "$UP" > verify.txt 2> verify.err || true
else "$INVAR" verify worldline.jsonl --binary "$BINARY" --model "$GGUF" > verify.txt 2> verify.err || true; fi
say "4. verify: $(tail -1 verify.txt)"; grep -q INDETERMINATE verify.txt && say "   (INDETERMINATE = your server did not reproduce itself; receipt intact; see FAQ)"
grep -qi "witness entries" verify.err && say "   (witness tier: structure, chain and signatures checked; a closed model cannot be re-executed by anyone)"

# 5. tamper a copy
python3 - <<'PY'
import json; e=json.loads(open("worldline.jsonl").readline()); e["output_text"]=(e.get("output_text") or "")+" [altered]"
open("tampered.jsonl","w").write(json.dumps(e)+"\n")
PY
if [ -n "$UP" ]; then "$INVAR" verify tampered.jsonl --upstream-url "$UP" > tamper.txt 2>/dev/null || true
else "$INVAR" verify tampered.jsonl --binary "$BINARY" --model "$GGUF" > tamper.txt 2>/dev/null || true; fi
say "5. tamper (altered output copy): $(tail -1 tamper.txt)"

# 6. summary
say "6. files in $OUT/: worldline.jsonl (your first receipt), verify.txt, tamper.txt$( [ -n "$UP" ] && echo ', triage-upstream.json')"
say "   next: send worldline.jsonl to someone outside your team with the verify command above; if they get the same verdict, the pilot's success line is one step away."
