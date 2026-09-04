#!/usr/bin/env sh
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
#
# INVAR installer — runs on the system you already have. Nothing flashed,
# nothing replaced: a private venv under ~/.invar and one `invar` command on
# your PATH. Uninstall = `rm -rf ~/.invar ~/.local/bin/invar`.
#
#   curl -fsSL https://get.anomly.com/invar.sh | sh
# or, from a downloaded release directory:  sh install.sh [wheel-or-sdist]
set -eu

PKG="${1:-anomly-invar}"           # release URL/path can be passed explicitly
INVAR_HOME="${INVAR_HOME:-$HOME/.invar}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

PY="$(command -v python3 || true)"
[ -n "$PY" ] || { echo "invar: python3 (>=3.10) is required"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || { echo "invar: python3 >= 3.10 required (found $($PY -V))"; exit 1; }

echo "-> creating $INVAR_HOME/venv"
"$PY" -m venv "$INVAR_HOME/venv"
"$INVAR_HOME/venv/bin/pip" install --quiet --upgrade pip
echo "-> installing $PKG"
"$INVAR_HOME/venv/bin/pip" install --quiet "$PKG"

mkdir -p "$BIN_DIR"
ln -sf "$INVAR_HOME/venv/bin/invar" "$BIN_DIR/invar"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "NOTE: add $BIN_DIR to your PATH to use 'invar' directly." ;;
esac

# a local llama.cpp runtime is needed for inference/verification re-execution
LLAMA="${INVAR_LLAMA_BIN:-$(command -v llama-cli || true)}"
if [ -n "$LLAMA" ]; then
  echo "-> llama.cpp runtime: $LLAMA"
else
  if command -v ollama >/dev/null 2>&1; then
    echo "-> Ollama found: $(command -v ollama) — 'invar serve --model <tag>' uses it"
  else
    echo "NOTE: no llama-cli or ollama found. Install Ollama (https://ollama.com) or"
    echo "      llama.cpp (https://github.com/ggml-org/llama.cpp), or set INVAR_LLAMA_BIN."
    echo "      Receipt STRUCTURE verification works without either; inference and"
    echo "      re-execution need one of them."
  fi
fi

echo ""
echo "INVAR installed."
echo "  invar verify <worldline.jsonl>            # free, forever"
echo "  invar serve --model <ollama-tag | model.gguf>   # receipted local endpoint"
echo "  invar license verify <license.invar>      # check a license offline"
