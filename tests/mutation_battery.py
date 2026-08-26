# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
mutation_battery.py — proves the unit suite has teeth. Injects one real bug per
security/correctness invariant, confirms unit_tests.py CATCHES it (exits non-zero),
then reverts. A SURVIVED mutant is a coverage hole: a bug the tests would miss.

Run:  python3 tests/mutation_battery.py     (exit 0 iff every mutant is caught)
Offline; touches only invar/*.py transiently and always restores the originals.
"""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = [sys.executable, os.path.join("tests", "unit_tests.py")]

# (file, description, old_substring, new_substring)
MUTANTS = [
    ("invar/crcore.py", "canonical JSON: don't sort keys",
     "sort_keys=True, separators", "sort_keys=False, separators"),
    ("invar/worldline.py", "build_entry: swap chain hash order",
     '(prev_chain + cert).encode()', '(cert + prev_chain).encode()'),
    ("invar/worldline.py", "verify: accept certificate mismatch",
     'if certificate_of(m) != e["certificate"]:',
     'if certificate_of(m) == e["certificate"] and False:'),
    ("invar/license.py", "check: ignore license expiry",
     "if lic.expired():", "if lic.expired() and False:"),
    ("invar/license.py", "check: trust any signing key",
     "if pub_b64 not in trusted:", "if pub_b64 not in trusted and False:"),
    ("invar/ledger.py", "verify_entry: accept forged certificate",
     'if certificate_of(m) != entry.get("certificate"):',
     'if certificate_of(m) == entry.get("certificate") and False:'),
    ("invar/serve.py", "serve: don't clamp max_tokens",
     'n = min(max(int(req.get("max_tokens") or 128), 1), 4096)',
     'n = max(int(req.get("max_tokens") or 128), 1)'),
    ("licensing/stripe_webhook.py", "webhook: accept any signature",
     "return hmac.compare_digest(exp, v1)", "return True"),
    ("licensing/stripe_webhook.py", "webhook: ignore timestamp tolerance",
     "if abs(time.time() - int(t)) > TOLERANCE:",
     "if abs(time.time() - int(t)) > TOLERANCE and False:"),
    ("licensing/stripe_webhook.py", "webhook: skip duplicate-delivery guard",
     "if os.path.exists(done_marker):",
     "if os.path.exists(done_marker) and False:"),
]


def main():
    caught = survived = 0
    for path, desc, old, new in MUTANTS:
        full = os.path.join(BASE, path)
        src = open(full).read()
        if old not in src:
            print(f"  [SKIP] {path}: anchor not found — {desc} (code changed?)")
            survived += 1
            continue
        open(full, "w").write(src.replace(old, new, 1))
        try:
            r = subprocess.run(SUITE, cwd=BASE, capture_output=True,
                               text=True, timeout=180)
        finally:
            open(full, "w").write(src)          # always revert
        if r.returncode != 0:
            caught += 1
            print(f"  [CAUGHT]   {path}: {desc}")
        else:
            survived += 1
            print(f"  [SURVIVED] {path}: {desc}  <-- TEST GAP")
    total = caught + survived
    print(f"\nmutants caught {caught}/{total}"
          + ("  ALL CAUGHT" if survived == 0 else f"  {survived} SURVIVED"))
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
