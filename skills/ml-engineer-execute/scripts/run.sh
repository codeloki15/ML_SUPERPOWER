#!/usr/bin/env bash
# Run a Python script under the ml-engineer venv.
# Usage: bash run.sh <path/to/script.py> [args...]
# Exits with the script's exit code. stdout/stderr passed through.

set -uo pipefail

VENV_ROOT="${CLAUDE_PLUGIN_DATA:-$HOME/.claude/ml-engineer}"
VENV="$VENV_ROOT/venv"

if [ ! -x "$VENV/bin/python" ]; then
    echo "error: venv not found at $VENV. Run setup_venv.sh first." >&2
    exit 127
fi

if [ "$#" -lt 1 ]; then
    echo "usage: run.sh <script.py> [args...]" >&2
    exit 64
fi

exec "$VENV/bin/python" "$@"
