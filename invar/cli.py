# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.cli — `invar verify <worldline.jsonl>`: verify a worldline end to end.

Checks every entry: certificate matches its canonical manifest, the hash chain
links from genesis, and (with --reexecute, the default) the pinned computation
is re-run and its output digest compared. Prompts come from the evidence text
stored beside each receipt (validated against the certified prompt digest
before use, so tampered evidence text cannot spoof a pass).

  invar verify worldline.jsonl [--binary llama-cli --model x.gguf]   # llama.cpp
  invar verify worldline.jsonl [--ollama-host URL]                    # Ollama
      (Ollama entries name their model tag in the receipt, so nothing else is
       needed when the server is reachable; --model overrides the tag)
  invar verify worldline.jsonl --no-reexecute        # structural + chain only
Exit code 0 = every entry ACCEPT; 1 = any REJECT.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from .attest import AttestationBinding
from .backends import (LLAMACPP_EXACT_PROFILE, LLAMACPP_PROFILE, OLLAMA_PROFILE,
                       LlamaCppBackend, OllamaBackend)
from .worldline import digest_bytes, verify_entries


def _profiles(path: str) -> set[str]:
    with open(path) as f:
        return {json.loads(line)["manifest"].get("profile", "") for line in f}


def main():
    # `invar serve ...` / `invar license ...` delegate to their modules with the
    # remaining argv, so each keeps its own argument surface.
    if len(sys.argv) > 1 and sys.argv[1] in ("serve", "license", "ledger"):
        sub = sys.argv.pop(1)
        if sub == "serve":
            from .serve import main as serve_main
            sys.argv[0] = "invar serve"
            return serve_main()
        if sub == "ledger":
            from .ledger import main as ledger_main
            sys.argv[0] = "invar ledger"
            return ledger_main()
        from .license import main as license_main
        sys.argv[0] = "invar license"
        return license_main()

    ap = argparse.ArgumentParser(
        prog="invar",
        description="INVAR — receipted local AI (verify | serve | license | ledger)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="verify a worldline receipt log")
    v.add_argument("worldline")
    v.add_argument("--binary", default=None,
                   help="llama.cpp binary for llama.cpp entries; or the ollama "
                        "binary to hash for Ollama entries (INVAR_OLLAMA_BIN)")
    v.add_argument("--model", default=None,
                   help="gguf path (llama.cpp entries) or Ollama tag override")
    v.add_argument("--ollama-host", default=None,
                   help="Ollama server for Ollama entries (default OLLAMA_HOST "
                        "or http://127.0.0.1:11434)")
    v.add_argument("--no-reexecute", action="store_true",
                   help="structural + chain checks only")
    v.add_argument("--attest", default=None,
                   help="attestation binding JSON the chain must be bound to")
    v.add_argument("--trust-key", action="append", default=None,
                   help="accept only signatures from this key_id (repeatable)")
    v.add_argument("--require-signature", action="store_true",
                   help="unsigned entries REJECT")
    v.add_argument("--spot-check", action="store_true",
                   help="exact-profile entries with a certified dump: re-execute challenged "
                        "lm_head rows from the pinned GGUF (needs --model)")
    v.add_argument("--rows", type=int, default=256, help="challenged rows per evaluation")
    v.add_argument("--units", action="store_true",
                   help="with --spot-check: also re-execute challenged rows of every layer's "
                        "FFN/attn-out matmuls (dump must have been made with --spot-check-units)")
    v.add_argument("--unit-rows", type=int, default=8, help="challenged rows per matmul unit")
    v.add_argument("--nonce", default=None, help="challenge nonce hex (default: fresh random)")
    at = sub.add_parser("attest", help="attestation bindings")
    ats = at.add_subparsers(dest="acmd", required=True)
    ab = ats.add_parser("bind", help="create a binding from platform evidence")
    ab.add_argument("--kind", required=True,
                    help="sev-snp-report | tdx-quote | nvidia-eta | tpm-quote | openpcc-bundle")
    ab.add_argument("--evidence", required=True, help="evidence file (raw bytes)")
    ab.add_argument("--verifier", default=None,
                    help="who verified it (e.g. snpguest, nvtrust, veraison)")
    ab.add_argument("--verdict", default=None, help="verifier's verdict/token file")
    ab.add_argument("--out", default="attest-binding.json")
    ap_ = ats.add_parser("collect-pcrs",
                         help="snapshot the TPM SHA-256 PCR bank from sysfs (unsigned evidence)")
    ap_.add_argument("--out", default="pcr-bank.json")
    ap_.add_argument("--pcrs", default="0-7", help="range or list, e.g. 0-7 or 0,2,4,7")
    aq = ats.add_parser("quote", help="SIGNED evidence: TPM 2.0 quote over PCRs with a nonce")
    aq.add_argument("--out", default="tpm-quote", help="bundle directory")
    aq.add_argument("--pcrs", default="sha256:0,1,2,4,7")
    av = ats.add_parser("check-quote", help="verify a quote bundle's signature + nonce")
    av.add_argument("bundle_dir")
    sc = sub.add_parser("scitt", help="SCITT-style Signed Statements (COSE_Sign1) over entries")
    scs = sc.add_subparsers(dest="scmd", required=True)
    ss = scs.add_parser("sign", help="emit one Signed Statement per entry")
    ss.add_argument("worldline")
    ss.add_argument("--signer", default=os.environ.get("INVAR_SIGNER", "software"),
                    help="software | tpm2 | tpm2:sha256:0,7 (same key store as serve)")
    ss.add_argument("--state-dir", default=os.environ.get("INVAR_STATE",
                    os.path.expanduser("~/.invar")))
    ss.add_argument("--issuer", required=True, help="e.g. did:web:yourco.example")
    ss.add_argument("--index", type=int, action="append", default=None,
                    help="only these entry indices (repeatable); default all")
    ss.add_argument("--out-dir", default="statements")
    sv = scs.add_parser("verify", help="verify a Signed Statement against a public key")
    sv.add_argument("statement")
    sv.add_argument("--pubkey", required=True, help="PEM public key (from /health or the signer)")
    sv.add_argument("--issuer", default=None)
    tg = sub.add_parser("tlog", help="transparency-log receipts (offline checks)")
    tgs = tg.add_subparsers(dest="tcmd", required=True)
    tc = tgs.add_parser("check", help="verify an inclusion receipt for a statement you hold")
    tc.add_argument("statement", help="COSE_Sign1 file")
    tc.add_argument("receipt", help="registration receipt JSON (from register / inclusion)")
    tcc = tgs.add_parser("consistency", help="verify a consistency proof between two heads")
    tcc.add_argument("old_head", help="JSON {tree_size, root}")
    tcc.add_argument("proof", help="JSON from /v1/tlog/consistency")
    a = ap.parse_args()

    if a.cmd == "tlog":
        from .tlog import check_receipt, check_consistency
        if a.tcmd == "check":
            with open(a.statement, "rb") as f:
                stb = f.read()
            ok = check_receipt(stb, json.load(open(a.receipt)))
        else:
            ok = check_consistency(json.load(open(a.old_head)), json.load(open(a.proof)))
        print("ACCEPT — inclusion proof verifies" if (ok and a.tcmd == "check")
              else "ACCEPT — log is append-only between the two heads" if ok
              else "REJECT — proof does not verify")
        sys.exit(0 if ok else 1)

    if a.cmd == "scitt" and a.scmd == "sign":
        from .hwsign import make_signer
        from .scitt import statements_for_worldline
        signer = make_signer(a.signer, a.state_dir)
        os.makedirs(a.out_dir, exist_ok=True)
        stmts = statements_for_worldline(a.worldline, signer, a.issuer, a.index)
        idx = a.index if a.index is not None else range(len(stmts))
        for i, st in zip(idx, stmts):
            with open(os.path.join(a.out_dir, f"entry-{i}.cose"), "wb") as f:
                f.write(st)
        with open(os.path.join(a.out_dir, "signer.pem"), "w") as f:
            f.write(signer.pubkey_pem)
        print(f"{len(stmts)} Signed Statement(s) -> {a.out_dir}/ (COSE_Sign1, "
              f"{signer.backend}, key {signer.key_id[:23]}…); public key in signer.pem")
        return
    if a.cmd == "scitt" and a.scmd == "verify":
        from .scitt import verify_statement
        with open(a.statement, "rb") as f:
            st = f.read()
        ok, info = verify_statement(st, open(a.pubkey).read(), a.issuer)
        if ok:
            print(f"ACCEPT — iss={info['iss']} sub={info['sub'][:30]}… alg={info['alg']} "
                  f"profile={info['manifest'].get('profile')}")
        else:
            print(f"REJECT — {info.get('why')}")
        sys.exit(0 if ok else 1)

    if a.cmd == "attest" and a.acmd == "quote":
        from .attest import collect_tpm_quote
        doc = collect_tpm_quote(a.out, a.pcrs)
        print(f"TPM quote bundle written to {a.out}/ (SIGNED by the AK; nonce {doc['nonce'][:16]}…)")
        print(f"bind with: invar attest bind --kind tpm-quote --evidence {a.out}/quote.msg "
              f"--verifier tpm2_checkquote --verdict {a.out}/bundle.json")
        return
    if a.cmd == "attest" and a.acmd == "check-quote":
        from .attest import verify_tpm_quote
        ok, why = verify_tpm_quote(a.bundle_dir)
        print(("ACCEPT" if ok else "REJECT") + " — " + why)
        sys.exit(0 if ok else 1)

    if a.cmd == "attest" and a.acmd == "collect-pcrs":
        from .attest import collect_pcr_bank
        sel: list[int] = []
        for part in a.pcrs.split(","):
            if "-" in part:
                lo, hi = part.split("-"); sel += list(range(int(lo), int(hi) + 1))
            else:
                sel.append(int(part))
        doc = collect_pcr_bank(a.out, tuple(sel))
        print(f"PCR bank written to {a.out} (UNSIGNED sysfs snapshot; a tpm2_quote is the signed form)")
        for k, v in doc["pcrs"].items():
            print(f"  PCR{k:>2} {v}")
        return

    if a.cmd == "attest":
        b = AttestationBinding(a.kind, a.evidence, a.verifier, a.verdict)
        b.save(a.out)
        print(f"binding written to {a.out}\n  kind={b.kind}\n  evidence={b.evidence_digest}"
              + (f"\n  verdict={b.verdict_digest} by {b.verifier}" if b.verdict_digest else "")
              + f"\n  genesis={b.genesis()}\nserve with: invar serve --attest {a.out} ...")
        return

    prompts: dict[str, str] = {}
    with open(a.worldline) as f:
        for line in f:
            e = json.loads(line)
            pt = e.get("prompt_text")
            if pt is not None:
                d = digest_bytes(pt.encode())
                if d == e["manifest"]["inputs"]["prompt"]:
                    prompts[d] = pt   # evidence text matches the certified digest

    reexec = not a.no_reexecute
    backends = {}
    if reexec:
        seen = _profiles(a.worldline)
        if LLAMACPP_PROFILE in seen or LLAMACPP_EXACT_PROFILE in seen:
            if a.binary and a.model:
                be = LlamaCppBackend(a.binary, a.model)
                backends[LLAMACPP_PROFILE] = LlamaCppBackend(a.binary, a.model,
                                                             profile=LLAMACPP_PROFILE)
                backends[LLAMACPP_EXACT_PROFILE] = LlamaCppBackend(
                    a.binary, a.model, profile=LLAMACPP_EXACT_PROFILE)
            else:
                print("llama.cpp entries: re-execution needs --binary and "
                      "--model; running structural checks on them only",
                      file=sys.stderr)
        if OLLAMA_PROFILE in seen:
            # one backend per model tag named in the receipts (--model overrides)
            override = a.model if (a.model and not os.path.exists(a.model)) else None
            backends[OLLAMA_PROFILE] = lambda tag: OllamaBackend(
                override or tag, host=a.ollama_host, binary=a.binary)
        if not backends:
            reexec = False

    binding = AttestationBinding.load(a.attest) if a.attest else None
    results = verify_entries(a.worldline, prompts, backends, reexecute=reexec,
                             binding=binding,
                             trusted_key_ids=set(a.trust_key) if a.trust_key else None,
                             require_signature=a.require_signature)
    if a.spot_check:
        import secrets
        from .spotcheck import dump_digest, verify_dump
        if not a.model:
            print("--spot-check needs --model <gguf>", file=sys.stderr)
            sys.exit(2)
        nonce = bytes.fromhex(a.nonce) if a.nonce else secrets.token_bytes(16)
        print(f"spot-check nonce {nonce.hex()} rows/eval {a.rows}")
        dumps_dir = a.worldline + ".dumps"
        entries = [json.loads(l) for l in open(a.worldline)]
        new_results = []
        for (i, ok, why), e in zip(results, entries):
            sc = e["manifest"].get("computation", {}).get("spot_check")
            if ok and sc:
                path = os.path.join(dumps_dir, sc["dump_digest"].split(":", 1)[1] + ".jsonl")
                if not os.path.exists(path):
                    ok, why = False, "spot-check dump missing"
                elif dump_digest(path) != sc["dump_digest"]:
                    ok, why = False, "spot-check dump digest differs from certified"
                else:
                    gobin = os.environ.get("INVAR_SPOTCHECK_BIN") or shutil.which("invar-spotcheck")
                    if gobin:                       # Go implementation, ~70x faster, same math
                        r = subprocess.run([gobin, "-gguf", a.model, "-dump", path, "-rows", str(a.rows),
                                            "-nonce", (nonce + i.to_bytes(4, "big")).hex()],
                                           capture_output=True, text=True, timeout=600)
                        sok = r.returncode == 0
                        swhy = (r.stdout.strip().splitlines() or ["(no output)"])[0].split("— ", 1)[-1] + " [go]"
                    else:
                        sok, swhy, _, _ = verify_dump(a.model, path, nonce + i.to_bytes(4, "big"), a.rows)
                    ok = ok and sok
                    why += "; spot-check: " + swhy
                    if ok and a.units:
                        from .spotcheck import verify_units
                        uok, uwhy, _, _, _ = verify_units(a.model, path, nonce + i.to_bytes(4, "big") + b"u",
                                                          rows=a.unit_rows)
                        ok = ok and uok
                        why += "; units: " + uwhy
            elif ok and e["manifest"].get("profile") == LLAMACPP_EXACT_PROFILE:
                why += "; spot-check: no dump certified for this entry"
            new_results.append((i, ok, why))
        results = new_results

    bad = 0
    for i, ok, why in results:
        print(f"entry {i}: {'ACCEPT' if ok else 'REJECT'} — {why}")
        bad += (not ok)
    print(f"{'ALL ACCEPT' if bad == 0 else f'{bad} REJECTED'} "
          f"({len(results)} entries)")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
