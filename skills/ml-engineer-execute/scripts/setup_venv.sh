#!/usr/bin/env bash
# Idempotent venv setup for ml-engineer plugin.
# Creates ${CLAUDE_PLUGIN_DATA}/venv (or ~/.claude/ml-engineer/venv as fallback)
# with pip upgraded. Packages are installed on demand by the execute/debug skills.
#
# Usage: bash setup_venv.sh
# Prints the venv path to stdout on success.

set -euo pipefail

VENV_ROOT="${CLAUDE_PLUGIN_DATA:-$HOME/.claude/ml-engineer}"
VENV="$VENV_ROOT/venv"

mkdir -p "$VENV_ROOT"

if [ -x "$VENV/bin/python" ]; then
    echo "venv already exists: $VENV" >&2
    echo "$VENV"
    exit 0
fi

# Pick the best available python3
PY=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "error: no python3 interpreter found on PATH" >&2
    exit 1
fi

echo "creating venv at $VENV using $PY ($($PY --version 2>&1))" >&2
"$PY" -m venv "$VENV"

# Upgrade pip; avoid installing packages eagerly. The orchestrator installs
# packages on demand when scripts raise ModuleNotFoundError.
"$VENV/bin/pip" install --quiet --upgrade pip

echo "venv ready: $VENV" >&2
echo "$VENV"
