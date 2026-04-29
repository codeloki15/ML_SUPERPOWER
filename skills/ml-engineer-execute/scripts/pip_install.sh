#!/usr/bin/env bash
# Install one or more packages into the ml-engineer venv.
# Usage: bash pip_install.sh pandas numpy scikit-learn

set -euo pipefail

VENV_ROOT="${CLAUDE_PLUGIN_DATA:-$HOME/.claude/ml-engineer}"
VENV="$VENV_ROOT/venv"

if [ ! -x "$VENV/bin/pip" ]; then
    echo "error: venv not found at $VENV. Run setup_venv.sh first." >&2
    exit 127
fi

if [ "$#" -lt 1 ]; then
    echo "usage: pip_install.sh <package> [more packages...]" >&2
    exit 64
fi

"$VENV/bin/pip" install "$@"
