# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""outbox_send.py — dispatch pending license emails from the webhook outbox.

Stub by design: v0 policy is MANUAL dispatch (Ry sees every founding customer).
When volume justifies automation, set SMTP_* env (or swap in a Resend/Postmark
call where marked) and cron this. Idempotent: a sent email gains a .sent marker.
"""
import os, smtplib, sys
from email.message import EmailMessage

OUTBOX = os.environ.get("OUTBOX_DIR", "outbox")
HOST, PORT = os.environ.get("SMTP_HOST"), int(os.environ.get("SMTP_PORT", "587"))
USER, PW = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
FROM = os.environ.get("SMTP_FROM", "licenses@anomly.com")

def pending():
    for f in sorted(os.listdir(OUTBOX)):
        if f.endswith(".email.txt") and not os.path.exists(
                os.path.join(OUTBOX, f + ".sent")):
            yield os.path.join(OUTBOX, f)

def main():
    if not HOST:
        n = len(list(pending()))
        print(f"{n} email(s) pending in {OUTBOX} — SMTP not configured; "
              "send manually or set SMTP_HOST/PORT/USER/PASS/FROM")
        return
    for path in pending():
        raw = open(path).read()
        to = raw.split("To: ", 1)[1].splitlines()[0]
        subject = raw.split("Subject: ", 1)[1].splitlines()[0]
        body = raw.split("\n", 2)[2]
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = FROM, to, subject
        msg.set_content(body)
        with smtplib.SMTP(HOST, PORT) as s:   # swap for Resend/Postmark here if preferred
            s.starttls(); s.login(USER, PW); s.send_message(msg)
        open(path + ".sent", "w").write("sent")
        print(f"sent -> {to}")

if __name__ == "__main__":
    main()
