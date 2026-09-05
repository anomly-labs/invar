#!/usr/bin/env sh
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
# One-command INVAR test run: fine-grained unit suite THEN the integration smokes.
#
# Everything runs OFFLINE by default: if you don't point it at a real model, the
# integration suite's inference sections use tests/fake_llama.py (a deterministic
# llama.cpp stand-in), so `sh tests/run_all.sh` gives full coverage on any machine.
# To exercise a REAL model instead, set both:
#   INVAR_TEST_MODEL=~/models/SmolLM2-135M-Instruct-q8_0.gguf \
#   INVAR_TEST_BINARY=$(command -v llama-cli) tests/run_all.sh
set -eu
cd "$(dirname "$0")/.."

echo "===== INVAR unit suite (offline, fake llama-cli) ====="
python3 tests/unit_tests.py

echo
echo "===== INVAR integration smokes ====="
CLEANUP=""
if [ -z "${INVAR_TEST_MODEL:-}" ] || [ -z "${INVAR_TEST_BINARY:-}" ]; then
    echo "(no real model set — provisioning the fake llama-cli so LLM sections run)"
    D=$(mktemp -d)
    CLEANUP="$D"
    cp tests/fake_llama.py "$D/llama-cli"
    chmod +x "$D/llama-cli"
    printf 'GGUF\000fake-weights' > "$D/model.gguf"
    INVAR_TEST_BINARY="$D/llama-cli"
    INVAR_TEST_MODEL="$D/model.gguf"
    export INVAR_TEST_BINARY INVAR_TEST_MODEL
fi
python3 tests/test_invar.py
rc=$?
[ -n "$CLEANUP" ] && rm -rf "$CLEANUP"

echo
echo "===== INVAR release-integrity gate ====="
python3 tests/test_release.py || rc=$?

echo
echo "===== INVAR installer + link-consistency gate ====="
python3 tests/test_install.py || rc=$?

echo
echo "===== INVAR stress + property-fuzz ====="
python3 tests/test_stress.py || rc=$?

echo
echo "===== INVAR deploy sanity (systemd + Dockerfile) ====="
python3 tests/test_deploy.py || rc=$?

echo
echo "===== INVAR CR-v0.1 conformance (byte-compat with the spec) ====="
python3 tests/test_cr_conformance.py || rc=$?

echo
echo "===== INVAR exact profile: detmath conformance, reference re-execution, tokenizer (skip without their optional deps/models) ====="
python3 tests/test_detmath.py || rc=$?
python3 tests/test_reexec_fixture.py || rc=$?
python3 tests/test_tokenizer_models.py || rc=$?
exit $rc
