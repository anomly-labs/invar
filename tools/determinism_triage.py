#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
# Licensed under the Apache License, Version 2.0 (same as the invar repository).
"""determinism_triage.py -- measure whether an OpenAI-compatible LLM server is reproducible,
and if not, where and how it diverges. Measurement only; standard library only; nothing is
changed on the server.

    determinism_triage.py --url http://host:port --model NAME [--prompt TEXT | --prompt-file F]
                          [--n 8] [--max-tokens 64] [--top-logprobs 5]
                          [--concurrent] [--sweep 0,64,256,1024] [--out report.json]
    determinism_triage.py --compare a.json b.json      # two setups, position by position

Axes (each is a separate, small experiment):
  repeat      the same request N times, one at a time: distinct completions, first divergence
              position, per-position top-1 agreement, per-position top-k *score* agreement
  ties        near-tie census from top_logprobs: at each position, the gap between the chosen
              token's logprob and the runner-up; a divergence at a position whose gap is large
              means the scores moved, not a tie-break (that is a reduction-order symptom)
  concurrent  the same N requests fired together: batch composition changes kernel shapes on
              many servers; compare against the serial run
  sweep       the same request behind prefixes of different lengths: reproducibility that
              comes and goes with prompt length is a kernel-selection boundary (M "islands")

Requires greedy decoding (temperature 0) and `logprobs` support with `top_logprobs`; servers
that ignore logprobs still get the text-level axes. Uses /v1/completions, falling back to
/v1/chat/completions (one user message) when the server only speaks chat."""
import argparse, hashlib, json, sys, time, threading, urllib.request, urllib.error

TRIAGE_VERSION = "0.4"

def get_json(url, path, timeout=10):
    """GET a JSON endpoint; None when absent. Returns (json, headers)."""
    try:
        req = urllib.request.Request(url.rstrip("/") + path, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            hdr = {k.lower(): v for k, v in r.headers.items()}
            try:
                return json.loads(r.read().decode()), hdr
            except Exception:
                return None, hdr
    except Exception:
        return None, {}

def server_info(url, model):
    """What the server says about itself, from the endpoints the common backends expose:
    /v1/models (everyone), /version (vLLM), /props (llama.cpp server: build, threads, flags),
    plus the Server header. Nothing here is trusted for the verdict; it labels the report."""
    info = {"url": url, "requested_model": model, "backend": "unknown", "version": None,
            "models": [], "props": None, "headers": {}}
    models, hdr = get_json(url, "/v1/models")
    info["headers"] = {k: hdr[k] for k in ("server", "x-powered-by") if k in hdr}
    if models and isinstance(models.get("data"), list):
        info["models"] = [{"id": m.get("id"), "owned_by": m.get("owned_by")} for m in models["data"]][:8]
    ver, _ = get_json(url, "/version")
    if isinstance(ver, dict) and "version" in ver:
        info["backend"], info["version"] = "vllm", ver["version"]
    props, _ = get_json(url, "/props")
    if isinstance(props, dict) and ("build_info" in props or "default_generation_settings" in props):
        info["backend"] = "llama.cpp"
        info["version"] = props.get("build_info")
        gs = props.get("default_generation_settings") or {}
        info["props"] = {"model_path": props.get("model_path"), "total_slots": props.get("total_slots"),
                         "n_ctx": gs.get("n_ctx"), "chat_template_present": bool(props.get("chat_template"))}
    if info["backend"] == "unknown":
        owned = {m.get("owned_by") for m in info["models"] if m.get("owned_by")}
        if "vllm" in owned: info["backend"] = "vllm"
        elif "llamacpp" in owned or "llama.cpp" in owned: info["backend"] = "llama.cpp"
        elif "openai" in owned: info["backend"] = "openai-compatible"
    return info

def post(url, body, timeout=600):
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

_USE_CHAT = {"v": None}   # None = unknown; True once /v1/completions has answered 404/405

def post_any(url, model, prompt, max_tokens, top_logprobs, seed):
    """/v1/completions first; servers that only speak chat (invar serve, some proxies) get the
    same request as a single user message with chat-style logprobs."""
    if not _USE_CHAT["v"]:
        try:
            return post(url, {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                              "seed": seed, "logprobs": top_logprobs}), False
        except urllib.error.HTTPError as e:
            if e.code not in (404, 405):
                raise
            _USE_CHAT["v"] = True
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
         "temperature": 0, "seed": seed, "logprobs": True, "top_logprobs": top_logprobs}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode()), True

def one_run(url, model, prompt, max_tokens, top_logprobs, seed):
    t0 = time.time()
    resp, chat = post_any(url, model, prompt, max_tokens, top_logprobs, seed)
    ch = resp["choices"][0]
    if chat and "message" in ch and "text" not in ch:
        ch = dict(ch); ch["text"] = (ch.get("message") or {}).get("content", "")
    lp = ch.get("logprobs") or {}
    if "content" in lp:                      # chat-style shape (llama.cpp server): content[{token, logprob, top_logprobs[]}]
        tokens = [e.get("token", "") for e in lp["content"]]
        chosen = [e.get("logprob") for e in lp["content"]]
        tops = [{t.get("token", ""): t.get("logprob") for t in (e.get("top_logprobs") or [])} for e in lp["content"]]
    else:                                     # legacy completions shape (vLLM, OpenAI): parallel arrays
        tokens = lp.get("tokens") or []
        chosen = lp.get("token_logprobs") or []
        tops = lp.get("top_logprobs") or []
    # per-position score fingerprint: the sorted (token, logprob) pairs of the top-k, exact floats
    fps, gaps = [], []
    for i, tl in enumerate(tops):
        if not tl:
            fps.append(None); gaps.append(None); continue
        items = sorted(tl.items(), key=lambda kv: (-kv[1], kv[0]))
        fps.append(hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()[:16])
        c = chosen[i] if i < len(chosen) and chosen[i] is not None else items[0][1]
        runner = next((v for t, v in items if v < c - 1e-12), None)
        if runner is None and len(items) > 1:
            runner = items[1][1]
        gaps.append(None if runner is None else c - runner)
    return {"text": ch.get("text", ""), "tokens": tokens, "fps": fps, "gaps": gaps,
            "finish": ch.get("finish_reason"), "secs": round(time.time() - t0, 3),
            "system_fingerprint": resp.get("system_fingerprint"), "served_model": resp.get("model"),
            "usage": resp.get("usage"), "receipted": "receipt" in resp}

def compare(runs):
    texts = [r["text"] for r in runs]
    distinct = len(set(texts))
    # first divergence position in tokens
    first = None
    L = min(len(r["tokens"]) for r in runs) if runs else 0
    for i in range(L):
        if len({r["tokens"][i] for r in runs}) > 1:
            first = i; break
    if first is None and len({len(r["tokens"]) for r in runs}) > 1:
        first = L
    # per-position agreement over the common prefix
    top1_agree = sum(1 for i in range(L) if len({r["tokens"][i] for r in runs}) == 1)
    have_fps = all(r["fps"] and all(f is not None for f in r["fps"][:L]) for r in runs) and L > 0
    score_agree = sum(1 for i in range(L) if len({r["fps"][i] for r in runs}) == 1) if have_fps else None
    # gap at the first divergence: large gap + divergence = scores moved, not a tie
    gap_at_div = None
    if first is not None and first < L and runs[0]["gaps"] and first < len(runs[0]["gaps"]):
        g = [r["gaps"][first] for r in runs if r["gaps"][first] is not None]
        gap_at_div = min(g) if g else None
    return {"runs": len(runs), "distinct_completions": distinct, "first_divergence_token": first,
            "common_prefix_tokens": L, "top1_agree_positions": top1_agree,
            "topk_score_agree_positions": score_agree, "min_gap_at_divergence": gap_at_div}

def tie_census(runs, thresholds=(1e-2, 1e-3, 1e-4, 1e-5)):
    gaps = [g for r in runs for g in r["gaps"] if g is not None]
    if not gaps:
        return None
    gaps_sorted = sorted(gaps)
    return {"positions": len(gaps), "min_gap": gaps_sorted[0], "median_gap": gaps_sorted[len(gaps) // 2],
            "below": {str(t): sum(1 for g in gaps if g < t) for t in thresholds}}

def compare_reports(pa, pb):
    """Position-by-position diff of two reports' reference runs: same prompt required."""
    A, B = json.load(open(pa)), json.load(open(pb))
    ra, rb = A.get("reference"), B.get("reference")
    if not ra or not rb:
        print("both reports need a 'reference' run (made with --out by triage >= 0.3)"); return 2
    if A["prompt_sha256"] != B["prompt_sha256"]:
        print("prompts differ (sha256 mismatch); nothing to compare"); return 2
    def who(R):
        sv = R.get("server", {})
        return f"{sv.get('backend')} {sv.get('version') or '?'} model {sv.get('served_model') or R.get('model')}" + (f" [{R['label']}]" if R.get("label") else "")
    print("A:", who(A)); print("B:", who(B))
    L = min(len(ra["tokens"]), len(rb["tokens"]))
    first_tok = next((i for i in range(L) if ra["tokens"][i] != rb["tokens"][i]), None)
    have = ra["fps"] and rb["fps"] and all(f is not None for f in ra["fps"][:L] + rb["fps"][:L])
    first_fp = next((i for i in range(L) if ra["fps"][i] != rb["fps"][i]), None) if have else None
    tok_agree = sum(1 for i in range(L) if ra["tokens"][i] == rb["tokens"][i])
    fp_agree = sum(1 for i in range(L) if ra["fps"][i] == rb["fps"][i]) if have else None
    print(f"common prefix {L} tokens: tokens agree {tok_agree}/{L}, top-k scores agree {fp_agree}/{L}" if have else
          f"common prefix {L} tokens: tokens agree {tok_agree}/{L} (no score fingerprints in one report)")
    print(f"first token difference at {first_tok}; first score difference at {first_fp}")
    if first_tok is not None and ra["gaps"] and first_tok < len(ra["gaps"]) and ra["gaps"][first_tok] is not None:
        print(f"A's margin at the token divergence: {ra['gaps'][first_tok]:.3g} nats"
              f" ({'not a tie' if ra['gaps'][first_tok] > 1e-3 else 'near-tie'})")
    if have and first_fp is not None and (first_tok is None or first_fp < first_tok):
        print(f"scores diverged {(first_tok if first_tok is not None else L) - first_fp} positions before the text did")
    print("verdict:", "BIT-IDENTICAL scores on the common prefix" if have and fp_agree == L else
          ("same text, different scores" if first_tok is None and have else "different"))
    return 0

def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--compare":
        sys.exit(compare_reports(sys.argv[2], sys.argv[3]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=None); ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--top-logprobs", type=int, default=5); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrent", action="store_true", help="also fire the N requests together")
    ap.add_argument("--sweep", default=None, help="comma-separated prefix lengths in tokens-ish (words) to prepend")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None, help="free-text name for this setup (GPU, driver, flags) for the report")
    a = ap.parse_args()
    prompt = a.prompt or (open(a.prompt_file).read() if a.prompt_file else
             "Explain in three sentences why floating-point summation depends on the order of the operands.")
    info = server_info(a.url, a.model)
    report = {"triage_version": TRIAGE_VERSION, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "label": a.label, "server": info, "client": {"python": sys.version.split()[0], "platform": sys.platform},
              "url": a.url, "model": a.model, "n": a.n, "max_tokens": a.max_tokens, "seed": a.seed,
              "top_logprobs": a.top_logprobs, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "axes": {}}
    print(f"target {a.url} model {a.model}; greedy, seed {a.seed}, {a.max_tokens} tokens, top_logprobs {a.top_logprobs}")
    print(f"server: backend {info['backend']}, version {info['version'] or '?'}, "
          f"models {[m['id'] for m in info['models']][:3]}{', props ' + json.dumps(info['props']) if info['props'] else ''}"
          + (f"; label: {a.label}" if a.label else ""))

    # -- repeat (serial)
    serial = [one_run(a.url, a.model, prompt, a.max_tokens, a.top_logprobs, a.seed) for _ in range(a.n)]
    rep = compare(serial); rep["tie_census"] = tie_census(serial)
    report["axes"]["repeat_serial"] = rep
    report["server"]["served_model"] = serial[0].get("served_model")
    if serial[0].get("receipted") and report["server"]["backend"] == "unknown":
        report["server"]["backend"] = "invar (receipted endpoint)"
    report["server"]["system_fingerprints"] = sorted({str(r.get("system_fingerprint")) for r in serial})
    if len(report["server"]["system_fingerprints"]) > 1:
        print(f"  note: system_fingerprint changed between runs: {report['server']['system_fingerprints']}")
    print(f"repeat (serial, n={a.n}): {rep['distinct_completions']} distinct completion(s); "
          f"first divergence at token {rep['first_divergence_token']}; "
          f"top-1 agree {rep['top1_agree_positions']}/{rep['common_prefix_tokens']}; "
          f"top-k scores agree {rep['topk_score_agree_positions']}/{rep['common_prefix_tokens']}")
    if rep["tie_census"]:
        tc = rep["tie_census"]
        print(f"  near-tie census over {tc['positions']} positions: min gap {tc['min_gap']:.3g}, median {tc['median_gap']:.3g}, "
              f"below 1e-3: {tc['below']['0.001']}, below 1e-4: {tc['below']['0.0001']}")
    if rep["first_divergence_token"] is not None and rep["min_gap_at_divergence"] is not None:
        print(f"  gap at the divergence position: {rep['min_gap_at_divergence']:.3g} "
              f"({'not a tie: the scores moved' if rep['min_gap_at_divergence'] > 1e-3 else 'near-tie'})")

    # -- concurrent
    if a.concurrent:
        results = [None] * a.n
        def w(i):
            try: results[i] = one_run(a.url, a.model, prompt, a.max_tokens, a.top_logprobs, a.seed)
            except Exception as e: results[i] = {"error": str(e)}
        ts = [threading.Thread(target=w, args=(i,)) for i in range(a.n)]
        [t.start() for t in ts]; [t.join() for t in ts]
        ok = [r for r in results if r and "error" not in r]
        if ok:
            conc = compare(ok + serial[:1])           # against one serial run as the reference
            conc["errors"] = a.n - len(ok)
            report["axes"]["concurrent_vs_serial"] = conc
            print(f"concurrent (n={len(ok)}) vs serial: {conc['distinct_completions']} distinct; "
                  f"first divergence at token {conc['first_divergence_token']}; "
                  f"top-k scores agree {conc['topk_score_agree_positions']}/{conc['common_prefix_tokens']}")

    # -- sweep
    if a.sweep:
        sw = {}
        filler = "The quick brown fox jumps over the lazy dog. "
        for L in [int(x) for x in a.sweep.split(",") if x.strip()]:
            p = (filler * (L // 9 + 1))[:max(0, L) * 5] + "\n\n" + prompt if L > 0 else prompt
            rs = [one_run(a.url, a.model, p, a.max_tokens, a.top_logprobs, a.seed) for _ in range(min(a.n, 4))]
            c = compare(rs); sw[str(L)] = c
            print(f"sweep prefix ~{L} words: {c['distinct_completions']} distinct of {len(rs)}; "
                  f"top-k scores agree {c['topk_score_agree_positions']}/{c['common_prefix_tokens']}")
        report["axes"]["prefix_sweep"] = sw

    # -- verdict line (measurement, not judgement)
    dist = rep["distinct_completions"]
    conc_d = report["axes"].get("concurrent_vs_serial", {}).get("distinct_completions")
    print("summary:", "reproducible on every axis measured" if dist == 1 and (conc_d in (None, 1)) else
          f"NOT reproducible: {dist} distinct serial completions" + (f", {conc_d} with concurrency" if conc_d else ""))
    if a.out:
        report["samples"] = [{"text": r["text"][:400], "tokens": len(r["tokens"]), "secs": r["secs"]} for r in serial]
        report["reference"] = {"prompt": prompt, "tokens": serial[0]["tokens"], "fps": serial[0]["fps"], "gaps": serial[0]["gaps"]}
        json.dump(report, open(a.out, "w"), indent=1)
        print("report ->", a.out)

if __name__ == "__main__":
    main()
