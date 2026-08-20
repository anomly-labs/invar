Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.

# INVAR Ledger — administrator guide (v0)

The Ledger is the licensed team plane: agents push worldline entries, the Ledger
**verifies every entry before storing it** (certificate recomputation + per-device
chain continuity) and exports certified chain-of-custody packets.

## Install & run
On any Linux host (self-hosted by design — your data never touches Anomly):
```
sh install.sh                       # same installer as the agent
sudo mkdir -p /etc/invar && sudo cp license.invar /etc/invar/
INVAR_LICENSE=/etc/invar/license.invar \
LEDGER_DIR=/var/lib/invar-ledger \
LEDGER_TOKEN=$(openssl rand -hex 24) \
invar ledger
```
Or use `deploy/invar-ledger.service` (systemd, DynamicUser, StateDirectory).
The Ledger refuses to start without a valid, unexpired `ledger` license.

## Exposure & TLS
Default bind is 127.0.0.1. To serve a fleet, set `HOST=0.0.0.0` and put a TLS
reverse proxy (Caddy/nginx) in front — the Ledger itself never terminates TLS.
All routes except /health require `Authorization: Bearer $LEDGER_TOKEN`;
distribute the token to agents via your secret manager. Rotate by restarting
with a new token and updating agents.

## Connecting agents
On each device:
```
LEDGER_URL=https://ledger.yourco.com LEDGER_TOKEN=... \
INVAR_DEVICE_ID=$(hostname) invar serve --model ...
```
Pushes are best-effort: a Ledger outage never blocks inference; the device's
local worldline remains the source of truth and re-pushes continue the chain.
A device that was rewritten locally will be REFUSED (422 chain break) — that
refusal is itself the audit signal.

## Exports (the deliverable)
```
curl -H "Authorization: Bearer $TOKEN" \
  "https://ledger.yourco.com/v1/export?device=alice-laptop" > custody-alice.json
```
The packet contains the entries, a per-entry verification report, the collector
attestation (licensee/tier), and a `packet_certificate` any third party can
recompute with the free verifier — no Anomly involvement needed.

## Operations
- **Backups:** `LEDGER_DIR` is plain append-only JSONL per device — snapshot it
  like any data directory. Restores keep chains valid (append-only).
- **Storage math:** ~1–3 KB per inference receipt (without evidence texts,
  digests only) — a 10-person team at 500 inferences/day ≈ 15 MB/day worst case.
- **Monitoring:** GET /health returns licensee; non-200 or a 422 spike from a
  device is your alert condition.
- **License renewal:** replace `/etc/invar/license.invar` and restart; expired
  licenses stop startup (running processes keep serving until restart).

## Known v0 limits (roadmap)
Single shared bearer token (per-device tokens in v1); no SSO/retention policies
yet; export is whole-device (time-range filters in v1). See docs/THREATMODEL.md
for the trust boundary — including that a fully-compromised device can fabricate
a plausible NEW history but cannot rewrite what the Ledger already holds.
