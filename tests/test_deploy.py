# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
test_deploy.py — sanity for the deploy artifacts a self-hosting customer copies
verbatim: the two systemd units and the container image. Offline, stdlib-only.
Catches drift like a unit that no longer launches the `invar` entrypoint, a
missing hardening directive, a shipped real token, or a Dockerfile that drops the
static-build / OpenMP-runtime flags llama.cpp needs.

  1 ledger unit   launches `invar ledger`, sets the required env, is hardened,
                  and ships only a CHANGE_ME token placeholder (never a real one)
  2 serve  unit   launches `invar serve --model ... --worldline ...`, hardened
  3 Dockerfile    multi-stage, static llama.cpp (BUILD_SHARED_LIBS=OFF), runtime
                  libgomp1, INVAR_LLAMA_BIN set, non-root USER, invar entrypoint

Run: python3 tests/test_deploy.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DEPLOY = os.path.join(ROOT, "deploy")

_fails = 0


def check(name, cond, detail=""):
    global _fails
    _fails += (not cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def parse_unit(path):
    """Minimal systemd-unit parser: {section: [(key, value), ...]} preserving
    duplicate keys (Environment= appears many times), comments/blank lines dropped."""
    sections, cur = {}, None
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1]
            sections.setdefault(cur, [])
        elif "=" in line and cur is not None:
            k, v = line.split("=", 1)
            sections[cur].append((k.strip(), v.strip()))
    return sections


def kv(sections, section, key):
    return [v for (k, v) in sections.get(section, []) if k == key]


def main():
    print("== deploy: systemd units + Dockerfile sanity ==")

    # 1. ledger unit
    lp = os.path.join(DEPLOY, "invar-ledger.service")
    L = parse_unit(lp)
    check("ledger unit has [Unit]/[Service]/[Install]",
          all(s in L for s in ("Unit", "Service", "Install")))
    exec_l = kv(L, "Service", "ExecStart")
    check("ledger ExecStart launches `invar ledger`",
          bool(exec_l) and exec_l[0].endswith("invar ledger")
          and os.path.basename(exec_l[0].split()[0]) == "invar", exec_l[0] if exec_l else "")
    env_l = {v.split("=", 1)[0] for v in kv(L, "Service", "Environment")}
    check("ledger sets INVAR_LICENSE + LEDGER_DIR + LEDGER_TOKEN",
          {"INVAR_LICENSE", "LEDGER_DIR", "LEDGER_TOKEN"} <= env_l, str(sorted(env_l)))
    check("ledger token is a placeholder, not a real secret",
          any(v == "LEDGER_TOKEN=CHANGE_ME" for v in kv(L, "Service", "Environment")))
    for d in ("NoNewPrivileges=true", "PrivateTmp=true"):
        k, val = d.split("=")
        check(f"ledger hardened: {d}", val in kv(L, "Service", k))
    check("ledger runs unprivileged (DynamicUser)", "true" in kv(L, "Service", "DynamicUser"))
    check("ledger restarts on failure", "on-failure" in kv(L, "Service", "Restart"))
    check("ledger WantedBy multi-user.target", "multi-user.target" in kv(L, "Install", "WantedBy"))

    # 2. serve unit
    sp = os.path.join(DEPLOY, "invar-serve.service")
    S = parse_unit(sp)
    exec_s = kv(S, "Service", "ExecStart")
    check("serve ExecStart launches `invar serve` with model+worldline",
          bool(exec_s) and "invar serve" in exec_s[0]
          and "--model" in exec_s[0] and "--worldline" in exec_s[0],
          exec_s[0] if exec_s else "")
    check("serve sets MODEL env",
          any(v.startswith("MODEL=") for v in kv(S, "Service", "Environment")))
    for d in ("NoNewPrivileges=true", "PrivateTmp=true"):
        k, val = d.split("=")
        check(f"serve hardened: {d}", val in kv(S, "Service", k))
    check("serve is a user service (WantedBy default.target)",
          "default.target" in kv(S, "Install", "WantedBy"))

    # 3. Dockerfile
    dp = os.path.join(ROOT, "Dockerfile")
    d = open(dp).read()
    check("Dockerfile is multi-stage (build + runtime)", d.count("FROM ") >= 2)
    check("Dockerfile builds llama.cpp static (BUILD_SHARED_LIBS=OFF)",
          "-DBUILD_SHARED_LIBS=OFF" in d)
    check("Dockerfile builds the llama-cli target", "llama-cli" in d)
    check("Dockerfile runtime installs libgomp1 (OpenMP runtime)", "libgomp1" in d)
    check("Dockerfile pins INVAR_LLAMA_BIN", "INVAR_LLAMA_BIN=" in d)
    check("Dockerfile installs the built invar wheel", "*.whl" in d and "pip install" in d)
    check("Dockerfile runs as non-root user", "useradd" in d and "USER invar" in d)
    check("Dockerfile entrypoint/cmd is invar", '"invar"' in d or "CMD [\"invar" in d)
    check("Dockerfile exposes serve+ledger ports", "8577" in d and "8579" in d)

    print(f"\n{'ALL PASS' if _fails == 0 else f'{_fails} FAILURES'} (deploy gate)")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
