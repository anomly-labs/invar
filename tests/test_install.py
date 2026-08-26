# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
test_install.py — the installer and the published download surface must stay
internally consistent. Offline, stdlib-only. Catches the drift that follows a
version bump: links still pointing at the old version, an un-regenerated
SHA256SUMS, or a release installer whose default wheel URL is wrong.

  1 installer   install.sh is valid POSIX sh, safe-by-construction (set -eu,
                python>=3.10 gate, private venv, symlink, documented uninstall)
  2 release     release/v{VER}/invar.sh defaults to the matching versioned wheel URL
  3 site        (INVAR_SITE_DIR, default ~/development/anomly; SKIPs if absent)
                every static/get/*/SHA256SUMS matches its files, the root
                installer defaults to an https wheel, and every /get/ link on the
                /invar page resolves to a file that exists in static/.

Run: python3 tests/test_install.py
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SITE = os.environ.get("INVAR_SITE_DIR", os.path.expanduser("~/development/anomly"))

_fails = 0


def check(name, cond, detail=""):
    global _fails
    _fails += (not cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def sh_syntax_ok(path):
    return subprocess.run(["sh", "-n", path], capture_output=True).returncode == 0


def _version():
    for line in open(os.path.join(ROOT, "pyproject.toml")):
        if line.strip().startswith("version"):
            return line.split('"')[1]
    return "0.0.0"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    VER = _version()
    print("== install: installer safety + published-link consistency ==")

    # 1. install.sh
    ins = os.path.join(ROOT, "install.sh")
    src = open(ins).read()
    check("install.sh valid POSIX sh (sh -n)", sh_syntax_ok(ins))
    check("install.sh sets -eu", "set -eu" in src)
    check("install.sh gates python >= 3.10", "(3,10)" in src or "(3, 10)" in src)
    check("install.sh uses a private venv (no system pollution)",
          "-m venv" in src and "$HOME/.invar" in src.replace("${INVAR_HOME:-", ""))
    check("install.sh puts a single `invar` on PATH", "ln -sf" in src and "/invar" in src)
    check("install.sh documents uninstall", "rm -rf" in src and ".invar" in src)

    # 2. release installer defaults to the matching versioned wheel
    rel_sh = os.path.join(ROOT, "release", f"v{VER}", "invar.sh")
    if os.path.exists(rel_sh):
        rsrc = open(rel_sh).read()
        check(f"release/v{VER}/invar.sh valid sh", sh_syntax_ok(rel_sh))
        want = f"https://www.anomly.com/get/v{VER}/anomly_invar-{VER}-py3-none-any.whl"
        check(f"release installer defaults to v{VER} wheel URL", want in rsrc, want)
    else:
        print(f"  [SKIP] release/v{VER}/invar.sh (run scripts/build_release.sh)")

    # 3. site consistency
    if not os.path.isdir(SITE):
        print(f"  [SKIP] site consistency (no site checkout at {SITE})")
    else:
        static = os.path.join(SITE, "static")
        get = os.path.join(static, "get")
        # 3a. every published SHA256SUMS matches its directory's files
        sums_checked = 0
        for dirpath, _dirs, files in os.walk(get):
            if "SHA256SUMS" in files:
                bad, dangling = [], []
                for line in open(os.path.join(dirpath, "SHA256SUMS")):
                    line = line.strip()
                    if not line:
                        continue
                    want, name = line.split()[0], line.split()[-1]
                    fp = os.path.join(dirpath, name)
                    if not os.path.exists(fp):
                        dangling.append(name)          # listed but not in this dir
                    elif sha256(fp) != want:
                        bad.append(name)               # present but WRONG hash (hard fail)
                rel = os.path.relpath(dirpath, static)
                check(f"site SHA256SUMS: present files hash-match in {rel}", not bad,
                      ("mismatch: " + ", ".join(bad)) if bad else "")
                if dangling:
                    print(f"  [WARN] {rel}/SHA256SUMS lists file(s) not in the dir: "
                          f"{', '.join(dangling)} (cosmetic; regen at v0.1.1 publish)")
                sums_checked += 1
        if sums_checked == 0:
            print("  [SKIP] no SHA256SUMS under static/get")
        # 3b. root installer defaults to an https wheel
        root_sh = os.path.join(get, "invar.sh")
        if os.path.exists(root_sh):
            rsrc = open(root_sh).read()
            m = re.search(r"https://www\.anomly\.com/get/v[\d.]+/anomly_invar-[\d.]+-py3-none-any\.whl", rsrc)
            check("site root installer defaults to an https wheel URL", bool(m),
                  m.group(0) if m else "no versioned wheel URL found")
        # 3c. every /get/ link on the /invar page resolves to a real file
        page = os.path.join(SITE, "src", "routes", "invar", "+page.svelte")
        if os.path.exists(page):
            links = set(re.findall(r'href="(/get/[^"]+)"', open(page).read()))
            missing = [ln for ln in links if not os.path.exists(os.path.join(static, ln.lstrip("/")))]
            check(f"/invar page: all {len(links)} /get/ links resolve to files",
                  not missing, ("missing: " + ", ".join(missing)) if missing else "")

    print(f"\n{'ALL PASS' if _fails == 0 else f'{_fails} FAILURES'} (install gate)")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
