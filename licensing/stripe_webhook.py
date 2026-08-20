# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
stripe_webhook.py — turns a completed Stripe Checkout into an issued INVAR license.
Stdlib + invar.license only (Stripe's webhook signature is HMAC-SHA256; no SDK).

Flow (matches licensing/SETUP.md):
  Stripe Payment Link checkout -> checkout.session.completed webhook -> this server
  verifies the Stripe-Signature header -> issues an Ed25519 license for the buyer's
  email (tier/seats from the price's lookup_key metadata) -> writes license +
  ready-to-send email into OUTBOX_DIR. Email dispatch is a separate step (SMTP or
  Resend key), so a webhook outage can never lose a purchase: the outbox is the
  queue and the audit trail.

Env: STRIPE_WEBHOOK_SECRET (whsec_...), ISSUING_KEY_PATH, OUTBOX_DIR, PORT (8578).
Price lookup_keys this understands (set them on the Stripe Prices):
  invar-founding-device  -> tier founding-device, perpetual, seats 1
  invar-ledger-seat      -> tier ledger, expires +1 month per renewal, seats = quantity
Run behind the reverse proxy that terminates TLS. Idempotent per session id.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from invar.license import issue  # noqa: E402

SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
KEY = os.environ.get("ISSUING_KEY_PATH", "")
OUTBOX = os.environ.get("OUTBOX_DIR", "outbox")
TOLERANCE = 300  # seconds, per Stripe docs
_ISSUE_LOCK = threading.Lock()  # serialize duplicate-delivery dedup

PLANS = {
    "invar-developer-monthly": {"tier": "ledger", "months": 1},   # personal Ledger, seats=1..5 devices
    "invar-enterprise-seat":   {"tier": "ledger", "months": 1},   # team Ledger, seats = quantity
    "invar-founding-device":   {"tier": "founding-device", "months": None},  # legacy/unused
}

EMAIL_TMPL = """To: {email}
Subject: Your ANOMLY INVAR license

Thanks for backing INVAR.

Attached (inline below) is your license file — save it as `license.invar`
next to your INVAR install, or pass its path via INVAR_LICENSE.

Install (on the system you already run — nothing flashed, nothing replaced):
  curl -fsSL https://www.anomly.com/get/invar.sh | sh

Verify your license offline any time:
  python3 -m invar.license verify license.invar

Your receipts, your machine, your proof.
— Anomly

----- license.invar -----
{license_json}
"""


def verify_sig(payload: bytes, header: str) -> bool:
    """Stripe-Signature: t=...,v1=... — HMAC-SHA256 over f'{t}.{payload}'."""
    try:
        parts = dict(kv.split("=", 1) for kv in header.split(","))
        t, v1 = parts["t"], parts["v1"]
    except Exception:
        return False
    if abs(time.time() - int(t)) > TOLERANCE:
        return False
    signed = f"{t}.".encode() + payload
    exp = hmac.new(SECRET.encode(), signed, hashlib.sha256).hexdigest()
    return hmac.compare_digest(exp, v1)


def handle_event(evt: dict) -> str | None:
    if evt.get("type") != "checkout.session.completed":
        return None
    sess = evt["data"]["object"]
    sid = sess["id"]
    done_marker = os.path.join(OUTBOX, f"{sid}.done")
    # concurrent duplicate deliveries (Stripe retries) could both pass an
    # unguarded exists() check and double-issue; serialize the whole issue path.
    with _ISSUE_LOCK:
        if os.path.exists(done_marker):
            return "duplicate"
        return _issue_locked(sess, sid, done_marker)


def _issue_locked(sess: dict, sid: str, done_marker: str) -> str | None:
    email = (sess.get("customer_details") or {}).get("email") or sess.get(
        "customer_email")
    if not email:
        return "no-email"
    # line items are not embedded in the event; the Payment Link's price lookup_key
    # is mirrored into session metadata by SETUP.md's link configuration.
    lookup = (sess.get("metadata") or {}).get("plan", "invar-founding-device")
    qty = int((sess.get("metadata") or {}).get("quantity", "1"))
    plan = PLANS.get(lookup)
    if plan is None:
        return f"unknown-plan:{lookup}"
    expires = None
    if plan["months"]:
        expires = (dt.date.today() + dt.timedelta(days=31 * plan["months"])
                   ).isoformat()
    blob = issue(KEY, email, plan["tier"], qty, expires)
    os.makedirs(OUTBOX, exist_ok=True)
    lic_json = json.dumps(blob, indent=1)
    with open(os.path.join(OUTBOX, f"{sid}.license.invar"), "w") as f:
        f.write(lic_json)
    with open(os.path.join(OUTBOX, f"{sid}.email.txt"), "w") as f:
        f.write(EMAIL_TMPL.format(email=email, license_json=lic_json))
    open(done_marker, "w").write(dt.datetime.now().isoformat())
    return f"issued:{plan['tier']}x{qty}:{email}"


class Hook(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        clen = int(self.headers.get("Content-Length", "0"))
        if clen > 1_000_000:
            self.send_response(413); self.end_headers(); return
        payload = self.rfile.read(clen)
        if not verify_sig(payload, self.headers.get("Stripe-Signature", "")):
            self.send_response(400); self.end_headers()
            self.wfile.write(b"bad signature")
            return
        try:
            result = handle_event(json.loads(payload))
            print(f"[webhook] {result}", flush=True)
            self.send_response(200); self.end_headers()
            self.wfile.write(b"ok")
        except Exception as e:                 # 500 -> Stripe retries; outbox is idempotent
            print(f"[webhook] ERROR {e}", file=sys.stderr, flush=True)
            self.send_response(500); self.end_headers()


if __name__ == "__main__":
    assert SECRET.startswith("whsec_"), "set STRIPE_WEBHOOK_SECRET"
    assert os.path.exists(KEY), "set ISSUING_KEY_PATH to the Ed25519 issuing key"
    port = int(os.environ.get("PORT", "8578"))
    print(f"INVAR license webhook on :{port} -> outbox {OUTBOX}", flush=True)
    host = os.environ.get("HOST", "127.0.0.1")   # behind the TLS proxy
    ThreadingHTTPServer((host, port), Hook).serve_forever()
