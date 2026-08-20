#!/usr/bin/env sh
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
# One-command INVAR test run. LLM-backed sections need a model + llama.cpp binary:
#   INVAR_TEST_MODEL=~/models/SmolLM2-135M-Instruct-q8_0.gguf \
#   INVAR_TEST_BINARY=$(command -v llama-cli) tests/run_all.sh
set -eu
cd "$(dirname "$0")/.."
exec python3 tests/test_invar.py
