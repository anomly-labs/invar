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
  echo "NOTE: no llama-cli found. Install llama.cpp (https://github.com/ggml-org/llama.cpp)"
  echo "      or set INVAR_LLAMA_BIN. Receipt STRUCTURE verification works without it;"
  echo "      inference and re-execution need it."
fi

echo ""
echo "INVAR installed."
echo "  invar verify <worldline.jsonl>            # free, forever"
echo "  invar serve --model <model.gguf>          # receipted local endpoint"
echo "  invar license verify <license.invar>      # check a license offline"
