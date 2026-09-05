# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.ledger — the INVAR Ledger collection plane (MVP). The licensed component:
agents push worldline entries; the collector verifies BEFORE storing (certificate
recomputation + per-device chain continuity), keeps append-only per-device logs,
and exports chain-of-custody packets that are themselves certified manifests.

  POST /v1/worldline/ingest   {device_id, entries:[...]}   (Bearer LEDGER_TOKEN)
  GET  /v1/export?device=ID   chain-of-custody packet (entries + verification
                              report + collector attestation, certificate over all)
  GET  /health

Env: INVAR_LICENSE (path; must verify, tier 'ledger', unexpired — the collector
refuses to start without it), LEDGER_DIR (storage), LEDGER_TOKEN (shared agent
transport token), PORT (8579), INVAR_TRUST_PUB (dev/test: extra trusted issuing
pubkey, base64 — production trust ships pinned in invar.license.TRUSTED_KEYS).

Verification-on-ingest means a Ledger never stores an entry it could not defend:
a certificate mismatch, a fork, or a gap is rejected at the door with the reason.
"""
from __future__ import annotations

import hashlib
import base64
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Server(ThreadingHTTPServer):
    # The OS default listen backlog (5) resets concurrent connects beyond it; a
    # fleet ingest endpoint must queue honest burst traffic, not drop it.
    request_queue_size = 64
from urllib.parse import parse_qs, urlparse

from .attest import AttestationBinding
from .crcore import certificate_of, digest_bytes
from .hwsign import verify_signature
from .tlog import TransparencyLog
from .license import TRUSTED_KEYS, check

GENESIS = "sha256:" + "0" * 64
_DEV_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _derived_genesis(host_attestation: dict) -> str:
    b = AttestationBinding(host_attestation["kind"], nonce=host_attestation.get("nonce"))
    b.evidence_digest = host_attestation.get("evidence_digest")
    return b.genesis()


class LedgerStore:
    """Per-device locking makes read-tip -> verify -> append ATOMIC under the
    threading server: without it, concurrent pushes for one device both verify
    against the same tip and both append — a duplicated/forked chain in the
    custody log (found by an 8-thread race test, 2026-08-20; test in suite)."""

    def __init__(self, root: str, trusted_key_ids: set | None = None, signer=None):
        self.signer = signer                    # Ledger's own key for SCITT statements
        os.makedirs(root, exist_ok=True)
        self.tlog = TransparencyLog(os.path.join(root, "tlog.b64"))   # statements registry
        # fleet-pinned device signing keys (LEDGER_TRUSTED_KEYS); None = accept any
        # key that verifies, but a signature that is PRESENT must always verify
        self.trusted_key_ids = trusted_key_ids
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._master = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _device_lock(self, device: str) -> threading.Lock:
        with self._master:
            return self._locks.setdefault(device, threading.Lock())

    def _path(self, device: str) -> str:
        assert _DEV_RE.match(device), "bad device id"
        return os.path.join(self.root, f"{device}.jsonl")

    def tip(self, device: str) -> str:
        t = GENESIS
        p = self._path(device)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    t = json.loads(line)["chain"]
        return t

    def verify_entry(self, entry: dict, prev: str) -> tuple[bool, str]:
        m = entry.get("manifest")
        if not isinstance(m, dict):
            return False, "no manifest"
        if certificate_of(m) != entry.get("certificate"):
            return False, "certificate mismatch"
        if m.get("prev_chain") != prev:
            # a device's FIRST entry may start at an attestation-bound genesis: the
            # binding is self-describing (evidence digest + nonce are certified in
            # the manifest), so recompute it rather than trusting the claim
            ha = m.get("computation", {}).get("host_attestation")
            bound_ok = (prev == GENESIS and ha and ha.get("kind", "none") != "none"
                        and _derived_genesis(ha) == m.get("prev_chain"))
            if not bound_ok:
                return False, f"chain break (expected prev {prev[:18]}…)"
            prev = m["prev_chain"]              # the bound genesis is the real prev
        want = "sha256:" + hashlib.sha256(
            (prev + entry["certificate"]).encode()).hexdigest()
        if entry.get("chain") != want:
            return False, "chain digest wrong"
        if entry.get("signature") is not None:
            ok, why = verify_signature(entry, self.trusted_key_ids)
            if not ok:
                return False, f"signature: {why}"
        return True, "ok"

    def ingest(self, device: str, entries: list[dict]) -> dict:
        with self._device_lock(device):
            return self._ingest_locked(device, entries)

    def _ingest_locked(self, device: str, entries: list[dict]) -> dict:
        prev = self.tip(device)
        accepted = 0
        for i, e in enumerate(entries):
            ok, why = self.verify_entry(e, prev)
            if not ok:
                return {"accepted": accepted, "rejected_at": i, "reason": why,
                        "tip": prev}
            with open(self._path(device), "a") as f:
                f.write(json.dumps(e, separators=(",", ":"), sort_keys=True) + "\n")
            prev = e["chain"]
            accepted += 1
        return {"accepted": accepted, "tip": prev}

    def export(self, device: str, collector: dict) -> dict:
        with self._device_lock(device):
            return self._export_locked(device, collector)

    def _export_locked(self, device: str, collector: dict) -> dict:
        entries, report, prev = [], [], GENESIS
        p = self._path(device)
        if os.path.exists(p):
            with open(p) as f:
                for i, line in enumerate(f):
                    e = json.loads(line)
                    ok, why = self.verify_entry(e, prev)
                    report.append({"entry": i, "ok": ok, "why": why})
                    entries.append(e)
                    prev = e["chain"]
        packet = {
            "invar_custody_packet": "1",
            "device": device,
            "tip": prev,
            "entry_count": len(entries),
            "verification": report,
            "collector": collector,
            "exported_unix": int(time.time()),
            "entries_digest": digest_bytes(
                json.dumps(entries, separators=(",", ":"),
                           sort_keys=True).encode()),
        }
        packet_cert = certificate_of(packet)
        return {"packet": packet, "entries": entries,
                "packet_certificate": packet_cert}


def make_handler(store: LedgerStore, token: str, collector: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            return self.headers.get("Authorization", "") == f"Bearer {token}"

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                return self._json(200, {"ok": True, "role": "invar-ledger",
                                        "licensee": collector["licensee"]})
            if u.path.startswith("/v1/tlog/"):
                if not self._authed():
                    return self._json(401, {"error": "auth"})
                q = parse_qs(u.query)
                try:
                    if u.path == "/v1/tlog/head":
                        return self._json(200, store.tlog.head())
                    if u.path == "/v1/tlog/inclusion":
                        return self._json(200, store.tlog.inclusion(int(q.get("index", ["-1"])[0])))
                    if u.path == "/v1/tlog/consistency":
                        return self._json(200, store.tlog.consistency(int(q.get("old_size", ["-1"])[0])))
                    if u.path == "/v1/tlog/leaf":
                        idx = int(q.get("index", ["-1"])[0])
                        return self._json(200, {"index": idx, "statement_b64":
                                                base64.b64encode(store.tlog.leaf(idx)).decode()})
                except (IndexError, ValueError):
                    return self._json(404, {"error": "no such index"})
                return self._json(404, {"error": "not found"})
            if u.path == "/v1/export":
                if not self._authed():
                    return self._json(401, {"error": "auth"})
                q = parse_qs(u.query)
                dev = (q.get("device") or [""])[0]
                if not _DEV_RE.match(dev):
                    return self._json(400, {"error": "bad device id"})
                packet = store.export(dev, collector)
                fmt = (q.get("format") or ["json"])[0]
                if fmt == "scitt":
                    # SCITT-style Signed Statement over the certified packet: payload =
                    # canonical packet manifest, CWT sub = packet certificate, signed by
                    # the Ledger's signer (LEDGER_SIGNER / --signer, TPM or software).
                    if store.signer is None:
                        return self._json(409, {"error": "ledger has no signer "
                                                "(start with --signer software|tpm2)"})
                    from .scitt import signed_statement
                    entry = {"manifest": packet["packet"],
                             "certificate": packet["packet_certificate"]}
                    st = signed_statement(entry, store.signer,
                                          collector.get("issuer", "invar-ledger"))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/cose; cose-type=\"cose-sign1\"")
                    self.send_header("Content-Length", str(len(st)))
                    self.send_header("X-Invar-Signer-Key-Id", store.signer.key_id)
                    if (q.get("register") or ["0"])[0] == "1":
                        # register the statement in the transparency log; the inclusion
                        # receipt travels in a header so the caller can verify offline
                        rcpt = store.tlog.append(st)
                        self.send_header("X-Invar-Tlog-Receipt",
                                         base64.b64encode(json.dumps(rcpt, separators=(",", ":"),
                                                                     sort_keys=True).encode()).decode())
                    self.end_headers()
                    self.wfile.write(st)
                    return
                return self._json(200, packet)
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/v1/tlog/register":
                if not self._authed():
                    return self._json(401, {"error": "auth"})
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    if n > 1_000_000:
                        return self._json(413, {"error": "too large"})
                    body = json.loads(self.rfile.read(n))
                    st = base64.b64decode(body["statement_b64"])
                    if not st or st[0] != 0xD2:
                        return self._json(400, {"error": "not a tagged COSE_Sign1"})
                    return self._json(200, store.tlog.append(st))
                except Exception as e:
                    return self._json(400, {"error": f"bad request: {e}"})
            if urlparse(self.path).path != "/v1/worldline/ingest":
                return self._json(404, {"error": "not found"})
            if not self._authed():
                return self._json(401, {"error": "auth"})
            try:
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 10_000_000:                    # 10 MB batch cap
                    return self._json(413, {"error": "request too large"})
                req = json.loads(self.rfile.read(clen))
                dev = req.get("device_id", "")
                if not _DEV_RE.match(dev):
                    return self._json(400, {"error": "bad device id"})
                entries = req.get("entries") or []
                if len(entries) > 1000:
                    return self._json(413, {"error": "max 1000 entries per push"})
                res = store.ingest(dev, entries)
                code = 200 if "rejected_at" not in res else 422
                self._json(code, res)
            except Exception as e:
                self._json(500, {"error": str(e)})
    return Handler


def store_from_env() -> "LedgerStore":
    """LEDGER_DIR, LEDGER_TRUSTED_KEYS (comma-separated key_ids), LEDGER_SIGNER
    (software | tpm2 | tpm2:sha256:0,7) with keys under INVAR_STATE (default ~/.invar)."""
    tk = os.environ.get("LEDGER_TRUSTED_KEYS", "").strip()
    signer = None
    if os.environ.get("LEDGER_SIGNER"):
        from .hwsign import make_signer
        state = os.environ.get("INVAR_STATE", os.path.expanduser("~/.invar"))
        os.makedirs(state, mode=0o700, exist_ok=True)
        signer = make_signer(os.environ["LEDGER_SIGNER"], state)
    return LedgerStore(os.environ.get("LEDGER_DIR", "ledger-data"),
                       trusted_key_ids={k.strip() for k in tk.split(",") if k.strip()} or None,
                       signer=signer)


def main():
    lic_path = os.environ.get("INVAR_LICENSE", "")
    trusted = TRUSTED_KEYS + ([os.environ["INVAR_TRUST_PUB"]]
                              if os.environ.get("INVAR_TRUST_PUB") else [])
    lic = check(lic_path, trusted) if lic_path else None
    if lic is None or lic.tier not in ("ledger", "pro"):
        print("invar-ledger: a valid 'ledger' license is required "
              "(set INVAR_LICENSE)", file=sys.stderr)
        sys.exit(2)
    token = os.environ.get("LEDGER_TOKEN", "")
    if not token:
        print("invar-ledger: set LEDGER_TOKEN (shared agent token)",
              file=sys.stderr)
        sys.exit(2)
    store = store_from_env()
    collector = {"licensee": lic.email, "tier": lic.tier, "seats": lic.seats,
                 "issuer": os.environ.get("LEDGER_ISSUER", f"invar-ledger:{lic.email}")}
    host = os.environ.get("HOST", "127.0.0.1")   # expose deliberately, behind TLS
    port = int(os.environ.get("PORT", "8579"))
    print(f"INVAR Ledger on {host}:{port}  licensee={lic.email} "
          f"dir={store.root}", flush=True)
    _Server((host, port),
            make_handler(store, token, collector)).serve_forever()


if __name__ == "__main__":
    main()
