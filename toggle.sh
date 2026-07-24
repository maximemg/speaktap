#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-}"

if [ -z "$UV_BIN" ]; then
    UV_BIN="$(command -v uv || true)"
fi
if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
fi
if [ -z "$UV_BIN" ]; then
    echo "uv is required; run $SCRIPT_DIR/install.sh first" >&2
    exit 1
fi

exec "$UV_BIN" run --project "$SCRIPT_DIR" speaktap toggle "$@"
