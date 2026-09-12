# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.certify — black-box determinism certification of any OpenAI-compatible endpoint.

Methodology (after vLLM issue #51187 / LockStep "certify"): a workload of S shared-prefix requests plus
F fillers is sent concurrently, repeated R times against the unchanged server after a discarded priming
pass; each request is then run alone. Observable per request: token strings, chosen-token logprob and
top-k alternatives, compared as exact doubles, all pairs. Greedy only (temperature 0, fixed seed).

The report places the endpoint on the certification ladder:
  L0 run-to-run determinism (single stream)     L1 batch invariance (solo == concurrent)
  L2 schedule/shape invariance (long fillers, all-pairs across repeats)
  L3/L4 identity with a run from another machine (`--compare`; L4 when the other machine is a different
        hardware class or OS, stated by the operator)   L5 third-party re-execution (needs INVAR receipts)
The certification itself is a receipt: a canonical manifest {endpoint, model, workload digest,
observations digest, ladder verdict, tool version} with its sha256 certificate, optionally signed.
"""
from __future__ import annotations
import hashlib, itertools, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import urllib.error, urllib.request

from .crcore import canonical_bytes, certificate_of

TOOL = "invar-certify/0.1"
SHARED_PREFIX = ("You are a careful assistant. Answer in one short paragraph without lists. "
                 "The question concerns the following program: def f(n): return n if n < 2 else f(n-1) + f(n-2). ")
SHARED_TAILS = ["What does f compute?", "What is the time complexity of f?", "Rewrite f iteratively.",
                "What does f(10) return?", "Why is f slow for large n?", "Add memoisation to f.",
                "Name one language feature that would speed f up.", "Is f tail recursive?",
                "What happens for negative n?", "Give a docstring for f.", "Explain f to a child.",
                "Compare f to a loop.", "What is f(0)?", "Where could f overflow?", "Is f pure?",
                "How would you test f?"]
FILLERS = ["Describe the colour of the sea at dusk in two sentences.", "List three uses of copper, in prose.",
           "What year did the first transatlantic telegraph cable open?", "Translate 'good morning' into French and German.",
           "Why do leaves change colour in autumn?", "What is the capital of Mongolia?", "Give one sentence about glaciers.",
           "Summarise the rules of tic-tac-toe.", "What is a haiku?", "How many legs does a spider have?",
           "Name a prime number between 90 and 100.", "What does a compiler do?", "Describe a pendulum's motion."]
LONG_PAD = ("The ledger records each transaction with its timestamp, counterparty, amount and a reference. "
            "Entries are appended in order and never edited; corrections are new entries that cite the old one. ")


def build_workload(n_shared: int, n_fillers: int, long_fillers: int = 0, long_words: int = 1200) -> list[str]:
    reqs = [SHARED_PREFIX + t for t in itertools.islice(itertools.cycle(SHARED_TAILS), n_shared)]
    fill = list(itertools.islice(itertools.cycle(FILLERS), n_fillers))
    pad = LONG_PAD * (long_words // 30 + 1)
    for i in range(min(long_fillers, len(fill))):
        fill[i] = pad[: long_words * 6] + " Now, ignoring the ledger text above: " + fill[i]
    return reqs + fill


def _http(method: str, url: str, headers: dict, body: Optional[dict], timeout: float):
    """stdlib HTTP (INVAR stays stdlib + cryptography): returns (status, json-or-None); 429 is retried with backoff."""
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(8):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode() or "null")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(float(e.headers.get("retry-after") or 0) or 4 * (attempt + 1)); continue
            return e.code, None
        except (urllib.error.URLError, TimeoutError, OSError):
            return 0, None
    return 429, None


def _one(url: str, key: str, model: str, prompt: str, max_tokens: int, top_logprobs: int, seed: int, timeout: float) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max_tokens, "seed": seed, "logprobs": True, "top_logprobs": top_logprobs, "stream": False}
    h = {"Content-Type": "application/json", "User-Agent": TOOL}
    if key:
        h["Authorization"] = f"Bearer {key}"
    status, j = _http("POST", f"{url}/chat/completions", h, body, timeout)
    if status != 200 or not j:
        return {"error": f"HTTP {status}"}
    ch = j["choices"][0]
    toks = [{"tok": t.get("token"), "lp": t.get("logprob"),
             "top": [[a.get("token"), a.get("logprob")] for a in (t.get("top_logprobs") or [])]}
            for t in ((ch.get("logprobs") or {}).get("content") or [])]
    text = ch["message"].get("content") or ""
    obs = json.dumps({"text": text, "toks": toks}, sort_keys=True)
    return {"text": text, "toks": toks, "obs_sha": hashlib.sha256(obs.encode()).hexdigest()}


def run_pass(reqs: list[str], workers: int, **kw) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda p: _one(prompt=p, **kw), reqs))


def first_diff(a: dict, b: dict):
    ta, tb = a.get("toks", []), b.get("toks", [])
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x["tok"] != y["tok"]:
            return ("token", i, f"{x['tok']!r} vs {y['tok']!r}")
        if x["lp"] != y["lp"]:
            d = abs((x["lp"] or 0.0) - (y["lp"] or 0.0))
            return ("logprob", i, f"{x['lp']!r} vs {y['lp']!r}", d)
        if [list(t) for t in x["top"]] != [list(t) for t in y["top"]]:
            return ("top_logprobs", i, "alternatives differ")
    if len(ta) != len(tb):
        return ("length", min(len(ta), len(tb)), f"{len(ta)} vs {len(tb)} tokens")
    if a.get("text") != b.get("text"):
        return ("text", -1, "text differs with identical tokens")
    return None


def l5_receipts(url: str, key: str, solo: list[dict], n_tail: int, reexec_binary: str = "", reexec_model: str = "",
                reexec_cross_deployment: bool = False, work_dir: str = "", log=print) -> Optional[dict]:
    """L5: does the endpoint publish INVAR receipts for what it just served, do they verify, and can a
    third party re-execute them? Fetches /v1/worldline/tail (INVAR serve), runs the structural
    verification (certificates, chain links, signatures) with reexecute=False, matches this run's solo
    outputs to receipted output digests, and, when a llama.cpp binary + GGUF are given, re-executes the
    matched entries with the INVAR verifier (the actual L5). Returns None when the endpoint has no receipts."""
    from .crcore import digest_bytes
    from .worldline import verify_entries, verify_worldline
    h = {"User-Agent": TOOL}
    if key:
        h["Authorization"] = f"Bearer {key}"
    status, j = _http("GET", f"{url}/worldline/tail?n={min(max(n_tail, 1), 100)}", h, None, 60)
    if status != 200 or not isinstance(j, dict):
        return None
    entries = j.get("entries") or []
    if not entries:
        return {"receipts": 0}
    path = os.path.join(work_dir or ".", "endpoint_worldline_tail.jsonl")
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    start_prev = (entries[0].get("manifest") or {}).get("prev_chain")
    structural = verify_entries(path, {}, {}, reexecute=False, start_prev=start_prev)
    n_ok = sum(1 for _, ok, _ in structural if ok)
    by_out = {}
    for idx, e in enumerate(entries):
        by_out.setdefault(((e.get("manifest") or {}).get("outputs") or {}).get("text"), idx)
    matched = {i: by_out[digest_bytes((o.get("text") or "").encode())] for i, o in enumerate(solo)
               if digest_bytes((o.get("text") or "").encode()) in by_out}
    out = {"receipts": len(entries), "chain_structural_accept": n_ok, "chain_structural_total": len(structural),
           "this_run_receipted": len(matched), "this_run_requests": len(solo),
           "profile": (entries[-1].get("manifest") or {}).get("profile"), "reexecuted": None}
    if reexec_binary and reexec_model and matched:
        prompts = {}
        for e in entries:
            pt = e.get("prompt_text")
            if pt is not None:
                prompts[digest_bytes(pt.encode())] = pt
        # re-execute the whole contiguous tail (chain intact from start_prev); judge the matched entries
        from .backends import LlamaCppBackend, LLAMACPP_PROFILE, LLAMACPP_EXACT_PROFILE
        # deployment pin (device, n_gpu_layers) defaults to what the receipts certify, as `invar verify` does
        comp0 = (entries[-1].get("manifest") or {}).get("computation") or {}
        kw = {}
        if reexec_cross_deployment:
            pass    # exact profile: re-execute on THIS verifier's deployment (CPU board vs GPU server); the verifier reports the differing pins
        elif comp0.get("device") is not None:
            dev = comp0["device"]; kw["device"] = "none" if dev == "none" else dev.split(":", 1)[0]   # description -> flag id
        if not reexec_cross_deployment and "n_gpu_layers" in comp0: kw["n_gpu_layers"] = comp0["n_gpu_layers"]
        backends = {LLAMACPP_PROFILE: LlamaCppBackend(reexec_binary, reexec_model, profile=LLAMACPP_PROFILE, **kw),
                    LLAMACPP_EXACT_PROFILE: LlamaCppBackend(reexec_binary, reexec_model, profile=LLAMACPP_EXACT_PROFILE, **kw)}
        res = verify_entries(path, prompts, backends, reexecute=True, start_prev=start_prev, cross_deployment=reexec_cross_deployment)
        want = set(matched.values())
        sel = [(i, ok, why) for i, ok, why in res if i in want]
        n_re = sum(1 for _, ok, why in sel if ok and "not re-executed" not in (why or "") and "no prompt text" not in (why or ""))
        out["reexecuted"] = {"accept": n_re, "of": len(sel), "cross_deployment": reexec_cross_deployment, "detail": [(i, ok, (why or "")[:120]) for i, ok, why in sel[:8]]}
        log(f"L5 re-execution: {n_re}/{len(sel)} of this run's receipts reproduce their output digest")
    return out


def _obs_digest(passes: list[list[dict]]) -> str:
    return hashlib.sha256(json.dumps([[o.get("obs_sha", "ERR") for o in p] for p in passes]).encode()).hexdigest()


def certify(url: str, model: str, *, api_key: str = "", shared: int = 6, fillers: int = 2, long_fillers: int = 0,
            repeats: int = 5, max_tokens: int = 24, top_logprobs: int = 5, seed: int = 0, workers: int = 0,
            timeout: float = 600.0, compare: Optional[dict] = None, compare_label: str = "",
            reexec_binary: str = "", reexec_model: str = "", reexec_cross_deployment: bool = False, work_dir: str = "", log=print) -> dict:
    url = url.rstrip("/")
    reqs = build_workload(shared, fillers, long_fillers)
    workers = workers or len(reqs)
    kw = dict(url=url, key=api_key, model=model, max_tokens=max_tokens, top_logprobs=top_logprobs, seed=seed, timeout=timeout)
    t0 = time.perf_counter(); run_pass(reqs, workers, **kw); log(f"priming pass {time.perf_counter()-t0:.1f}s")
    reps = []
    for r in range(repeats):
        t0 = time.perf_counter(); obs = run_pass(reqs, workers, **kw); reps.append(obs)
        log(f"repeat {r}: {time.perf_counter()-t0:.1f}s, errors {sum('error' in o for o in obs)}")
    t0 = time.perf_counter(); solo = run_pass(reqs, 1, **kw); log(f"solo pass {time.perf_counter()-t0:.1f}s")
    n = len(reqs)
    errors = sum("error" in o for p in reps + [solo] for o in p)
    # L0/L2: all pairs across repeats (L0 = the single-stream requests; here every request is a stream)
    pair_diffs, worst, token_div = [], 0.0, False
    for p, q in itertools.combinations(range(repeats), 2):
        d = [(i, first_diff(reps[p][i], reps[q][i])) for i in range(n)]
        d = [(i, x) for i, x in d if x]
        if d:
            pair_diffs.append({"pair": [p, q], "n_requests": len(d), "first": [d[0][0], list(d[0][1][:3])]})
            for _, x in d:
                if x[0] == "token":
                    token_div = True
                if x[0] == "logprob":
                    worst = max(worst, x[3])
    distinct = len({tuple(o.get("obs_sha", "ERR") for o in p) for p in reps})
    # L1: solo == concurrent (against repeat 0)
    solo_diffs = [(i, first_diff(solo[i], reps[0][i])) for i in range(n)]
    solo_diffs = [(i, list(x[:3])) for i, x in solo_diffs if x]
    # L3/L4: compare with another machine's solo observations on the same workload
    cross = None
    if compare is not None:
        if compare.get("requests") != reqs:
            cross = {"comparable": False, "why": "different workload"}
        else:
            prev = compare.get("solo") or compare["repeats"][0]
            xd = [(i, first_diff(prev[i], solo[i])) for i in range(n)]
            xd = [(i, list(x[:3])) for i, x in xd if x]
            cross = {"comparable": True, "identical": not xd, "n_differ": len(xd), "first": xd[0] if xd else None,
                     "label": compare_label or compare.get("label", ""),
                     "positions": sum(len(o["toks"]) for o in solo)}
    l5 = l5_receipts(url, api_key, solo, n_tail=n * (repeats + 2), reexec_binary=reexec_binary, reexec_model=reexec_model,
                     reexec_cross_deployment=reexec_cross_deployment, work_dir=work_dir, log=log)
    l5_verdict = None
    if l5 and l5.get("receipts"):
        if l5.get("reexecuted"):
            l5_verdict = l5["reexecuted"]["accept"] == l5["reexecuted"]["of"] and l5["this_run_receipted"] == n
        else:
            l5_verdict = None      # receipts present and structurally sound, but not re-executed here
    ladder = {
        "L0_run_to_run_deterministic": distinct == 1 and errors == 0,
        "L1_batch_invariant": not solo_diffs and errors == 0,
        "L2_schedule_shape_invariant": (distinct == 1 and not solo_diffs and errors == 0) if long_fillers else None,
        "L3_L4_cross_machine_identical": (cross["identical"] if cross and cross["comparable"] else None),
        "L5_third_party_reexecutable": l5_verdict,
    }
    report = {
        "tool": TOOL, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "endpoint": url, "model": model,
        "workload": {"n": n, "shared": shared, "fillers": fillers, "long_fillers": long_fillers, "repeats": repeats,
                     "workers": workers, "max_tokens": max_tokens, "top_logprobs": top_logprobs, "seed": seed,
                     "digest": hashlib.sha256(json.dumps(reqs).encode()).hexdigest()},
        "results": {"distinct_workload_outputs": distinct, "repeat_pairs_differ": len(pair_diffs),
                    "repeat_pairs_total": repeats * (repeats - 1) // 2, "max_logprob_delta": worst,
                    "token_divergence": token_div, "solo_vs_concurrent_differ": len(solo_diffs), "errors": errors,
                    "cross": cross, "l5": l5, "observations_digest": _obs_digest(reps + [solo])},
        "ladder": ladder,
    }
    manifest = {"profile": "invar-certify-v0", "tool": TOOL, "endpoint": url, "model": model,
                "workload_sha256": report["workload"]["digest"], "observations_sha256": report["results"]["observations_digest"],
                "ladder": ladder, "utc": report["utc"]}
    report["certificate"] = certificate_of(manifest)
    report["manifest"] = manifest
    report["_observations"] = {"requests": reqs, "repeats": reps, "solo": solo, "label": ""}   # the CLI fills in this machine's label
    return report


def _l5_cell(l5, verdict) -> str:
    if not l5:
        return "no receipts at this endpoint (not an INVAR serve): cannot be established"
    if not l5.get("receipts"):
        return "endpoint publishes a worldline but it is empty"
    base = (f"{l5['receipts']} receipts fetched; chain/certificates structurally ACCEPT {l5['chain_structural_accept']}/{l5['chain_structural_total']}; "
            f"{l5['this_run_receipted']}/{l5['this_run_requests']} of this run's outputs receipted (profile `{l5.get('profile')}`)")
    if l5.get("reexecuted"):
        re = l5["reexecuted"]
        return ("**pass**" if verdict else "**fail**") + f" — re-executed {re['accept']}/{re['of']} matched receipts with the INVAR verifier; " + base
    return "receipts present, not re-executed here (pass --reexec-binary and --reexec-model); " + base


def render_md(rep: dict) -> str:
    r, w, L = rep["results"], rep["workload"], rep["ladder"]
    def yn(v):
        return "not tested" if v is None else ("**pass**" if v else "**fail**")
    reached = [k for k, v in L.items() if v is True]
    highest = reached[-1].split("_")[0] if reached else "none"
    first_fail = next((k.split("_")[0] for k, v in L.items() if v is False), None)
    verdict = f"**Highest rung passed: {highest}**" + (f"; first failure: {first_fail}" if first_fail else "; no failures among the rungs tested")
    md = [f"# Determinism certification — `{rep['endpoint']}` model `{rep['model']}` ({rep['utc']})", "", verdict, "",
          f"{w['n']} requests ({w['shared']} shared-prefix + {w['fillers']} fillers, {w['long_fillers']} long), {w['workers']} concurrent, "
          f"{w['repeats']} repeats after a priming pass, greedy, seed {w['seed']}, max_tokens {w['max_tokens']}, top_logprobs {w['top_logprobs']}.", "",
          "| level | property | result |", "|---|---|---|",
          f"| L0 | run-to-run deterministic | {yn(L['L0_run_to_run_deterministic'])} — {r['distinct_workload_outputs']} distinct workload output(s) in {w['repeats']} repeats; {r['repeat_pairs_differ']}/{r['repeat_pairs_total']} pairs differ; max logprob delta {r['max_logprob_delta']:.3e}; token divergence {'yes' if r['token_divergence'] else 'no'} |",
          f"| L1 | batch invariant (solo == concurrent) | {yn(L['L1_batch_invariant'])} — {w['n'] - r['solo_vs_concurrent_differ']}/{w['n']} identical |",
          f"| L2 | schedule/shape invariant (long fillers) | {yn(L['L2_schedule_shape_invariant'])} |",
          f"| L3/L4 | identical to a run from another machine | {yn(L['L3_L4_cross_machine_identical'])}" +
          (f" — vs `{r['cross'].get('label','')}`: {'identical' if r['cross'].get('identical') else str(r['cross'].get('n_differ')) + ' differ, first ' + str(r['cross'].get('first'))} over {r['cross'].get('positions')} positions" if r["cross"] and r["cross"].get("comparable") else "") + " |",
          "| L5 | third-party re-executable | " + _l5_cell(r.get("l5"), L["L5_third_party_reexecutable"]) + " |",
          "", f"errors: {r['errors']}; observations digest `{r['observations_digest'][:16]}…`; certificate `{rep['certificate']}`"]
    return "\n".join(md) + "\n"
