# ML Engineer (Claude Code Plugin)

A local-only ML engineer assistant for Claude Code. Plans, writes, executes, and debugs Python data-science / ML tasks in an isolated venv on your machine. No cloud services beyond Claude Code itself.

## What's in here

```
ML_Engineer/
├── .claude-plugin/
│   └── plugin.json
├── agents/
│   └── ml-engineer.md              # subagent that drives the plan→write→execute→debug loop
├── skills/
│   ├── ml-engineer-plan/           # produces a checkbox TODO plan
│   ├── ml-engineer-write-code/     # writes self-contained Python scripts
│   ├── ml-engineer-execute/        # runs scripts under the local venv
│   │   └── scripts/
│   │       ├── setup_venv.sh
│   │       ├── run.sh
│   │       └── pip_install.sh
│   └── ml-engineer-debug/          # diagnoses failures, returns patched scripts
└── README.md
```

## Install (local development)

```bash
# From any Claude Code session in this directory:
/plugin marketplace add /Users/lokesh/Desktop/code/Bio_hacking/ML_Engineer
/plugin install ml-engineer
```

(Or load directly via `claude --plugin-dir /Users/lokesh/Desktop/code/Bio_hacking/ML_Engineer`.)

## How it works

1. You ask: *"Analyze this CSV and find correlations."*
2. The `ml-engineer` subagent activates and invokes `ml-engineer-plan` → shows you a TODO list.
3. You approve.
4. For each step: `ml-engineer-write-code` writes a script to `./newton_workdir/<timestamp>/step_N.py`.
5. `ml-engineer-execute` runs it under the venv at `~/.claude/ml-engineer/venv` (creates it on first use after asking for approval).
6. On failure, `ml-engineer-debug` returns a patched script and the loop retries (up to 3 times per step).

## Venv

- Location: `${CLAUDE_PLUGIN_DATA}/venv`, falls back to `~/.claude/ml-engineer/venv`.
- Created on first use, with one-time user approval.
- Empty by default — packages are installed on demand when scripts raise `ModuleNotFoundError` (with per-install user approval).
- Reused across all sessions.

## Workdir

Each session writes to `./newton_workdir/<UTC-timestamp>/` in the user's current directory:

```
newton_workdir/2026-04-29T14-22-09Z/
├── step_1_load.py
├── step_2_explore.py
├── output_summary.txt
└── charts/
    ├── correlation_matrix.png
    └── target_distribution.png
```

The workdir is the user's — the plugin never deletes it.

## Hard constraints (enforced by the skills)

- No `plt.show()`, `input()`, web servers, infinite loops, or destructive file ops in generated code.
- No system Python, no global pip, no `sudo`.
- Code only runs from inside `newton_workdir/<timestamp>/`.
- Charts always saved to `charts/` and announced via `print("Chart saved as ...")`.

## License

MIT
