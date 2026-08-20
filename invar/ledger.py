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
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .crcore import certificate_of, digest_bytes
from .license import TRUSTED_KEYS, check

GENESIS = "sha256:" + "0" * 64
_DEV_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class LedgerStore:
    """Per-device locking makes read-tip -> verify -> append ATOMIC under the
    threading server: without it, concurrent pushes for one device both verify
    against the same tip and both append — a duplicated/forked chain in the
    custody log (found by an 8-thread race test, 2026-08-20; test in suite)."""

    def __init__(self, root: str):
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
            return False, f"chain break (expected prev {prev[:18]}…)"
        want = "sha256:" + hashlib.sha256(
            (prev + entry["certificate"]).encode()).hexdigest()
        if entry.get("chain") != want:
            return False, "chain digest wrong"
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
            if u.path == "/v1/export":
                if not self._authed():
                    return self._json(401, {"error": "auth"})
                dev = (parse_qs(u.query).get("device") or [""])[0]
                if not _DEV_RE.match(dev):
                    return self._json(400, {"error": "bad device id"})
                return self._json(200, store.export(dev, collector))
            self._json(404, {"error": "not found"})

        def do_POST(self):
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
    store = LedgerStore(os.environ.get("LEDGER_DIR", "ledger-data"))
    collector = {"licensee": lic.email, "tier": lic.tier, "seats": lic.seats}
    host = os.environ.get("HOST", "127.0.0.1")   # expose deliberately, behind TLS
    port = int(os.environ.get("PORT", "8579"))
    print(f"INVAR Ledger on {host}:{port}  licensee={lic.email} "
          f"dir={store.root}", flush=True)
    ThreadingHTTPServer((host, port),
                        make_handler(store, token, collector)).serve_forever()


if __name__ == "__main__":
    main()
