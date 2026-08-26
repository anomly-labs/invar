<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
# INVAR tests

The whole battery runs **offline** — no model, GPU, or network. A deterministic
llama.cpp stand-in (`fake_llama.py`) covers the inference / verification /
re-execution paths, so what would otherwise need a real model runs in CI.

```sh
sh tests/run_all.sh
```

Only dependency: `cryptography` (already required by the agent). Everything else
is the Python standard library. Exit code is non-zero if any gate fails.

## The gates

| File | What it proves |
|------|----------------|
| `unit_tests.py` | Fine-grained coverage of every module (crcore, worldline, license, ledger, serve, cli, webhook) — 137 assertions, ~97% line coverage, including every error branch and HTTP validation path. |
| `test_invar.py` | Integration smokes: receipted inference end-to-end, license lifecycle, Stripe webhook, ledger ingest/export, full serve→ledger→export stack. |
| `test_stress.py` | 25 parallel completions must yield one un-forked worldline (serve + append locks); 800-trial property-fuzz of the CR canonicalization (key-order invariance, determinism, unicode, NaN/±Inf rejection). |
| `test_release.py` | The downloadable artifacts: checksums, wheel metadata + entry point, shipped bytes run, and the wheel is **not stale** vs source. Skips if no release dir. |
| `test_install.py` | `install.sh` is safe POSIX sh; the release installer targets the right wheel URL; site `SHA256SUMS` and `/get/` links stay consistent. Skips the site half if absent. |
| `test_deploy.py` | The systemd units launch the `invar` entrypoint and are hardened; the Dockerfile is multi-stage, static-builds llama.cpp, and ships the OpenMP runtime. |
| `mutation_battery.py` | Injects one real bug per security/correctness invariant and confirms the unit suite **catches** it (10/10). A surviving mutant is a coverage hole. |

## Running individual pieces

```sh
python3 tests/unit_tests.py          # fastest, run this while developing
python3 tests/mutation_battery.py    # prove the suite still has teeth
python3 tests/test_stress.py         # concurrency + fuzz
```

## Against a real model (optional)

The integration suite uses the fake runtime by default. To exercise real
llama.cpp instead:

```sh
INVAR_TEST_MODEL=~/models/your.gguf \
INVAR_TEST_BINARY="$(command -v llama-cli)" \
python3 tests/test_invar.py
```

## Coverage

```sh
pip install coverage
coverage run tests/unit_tests.py && coverage report -m
```
