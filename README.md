# ML Engineer (Claude Code Plugin)

A local-only data-science / ML / quant assistant for Claude Code. Plans, researches, writes, executes, verifies, debugs, and reviews Python work in an isolated venv on your machine. Domain-agnostic — works for ML, finance, healthcare, drug discovery, retail, forecasting, ops research, and any quantitative discipline. No cloud services beyond Claude Code itself.

## What's in here

```
ML_Engineer/
├── .claude-plugin/
│   └── plugin.json
├── agents/
│   └── ml-engineer.md              # orchestrator subagent — drives the loop
├── skills/
│   ├── ml-engineer-research/       # WebSearch + WebFetch, returns conclusions, no citations
│   ├── ml-engineer-decide/         # evidence → recommendation with approval gate
│   ├── ml-engineer-hypothesis/     # falsifiable, testable hypotheses
│   ├── ml-engineer-plan/           # checkbox TODO plan
│   ├── ml-engineer-cv-design/      # picks CV scheme by data shape (Stratified / Group / walk-forward / binned-stratified)
│   ├── ml-engineer-pick-metric/    # locks evaluation metric before training
│   ├── ml-engineer-write-code/     # Python scripts; Layout A (one-off) or Layout B (project-style for training)
│   ├── ml-engineer-execute/        # runs scripts under the local venv
│   │   └── scripts/
│   │       ├── setup_venv.sh
│   │       ├── run.sh
│   │       └── pip_install.sh
│   ├── ml-engineer-verify/         # per-step verification (Iron Law: no completion claim without fresh evidence)
│   ├── ml-engineer-debug/          # 4-phase root-cause debugging, 3-failures escape hatch
│   └── ml-engineer-review/         # end-of-task critique, severity-tagged findings
└── README.md
```

## Install (local development)

```bash
# From any Claude Code session in this directory:
/plugin marketplace add /Users/lokesh/Desktop/code/Bio_hacking/ML_Engineer
/plugin install ml-engineer
```

Or directly:

```bash
claude --plugin-dir /Users/lokesh/Desktop/code/Bio_hacking/ML_Engineer
```

## How it works

The `ml-engineer` subagent activates whenever you describe a quantitative task and drives a deterministic loop:

```
research → decide → plan → write-code → execute → verify → debug (if needed) → review
```

It runs **autonomously**. You can interrupt at any time, but the agent does not pause for plan approval — absence of interruption is consent. The first time the venv is needed it asks once, then is silent on subsequent runs.

Per-step:

1. Writes a script to `./newton_workdir/<UTC-timestamp>/step_N_<name>.py`.
2. Runs it under the venv at `${CLAUDE_PLUGIN_DATA}/venv` (falls back to `~/.claude/ml-engineer/venv`).
3. On failure, `ml-engineer-debug` performs root-cause analysis (not symptom patching) and returns a corrected script. After 3 failed fixes on the same step, it stops patching and audits the data pipeline / assumptions / architecture instead.
4. On success, `ml-engineer-verify` runs a separate check via a different code path — exit-0 is not verification.

Before declaring the whole task complete, `ml-engineer-review` runs a fresh-eyes critique covering plan-vs-result drift, methodological soundness (walk-forward CV, transaction costs, multiple-testing correction, scaffold splits, assumption checks), reproducibility, and honesty of the result.

## Venv

- Location: `${CLAUDE_PLUGIN_DATA}/venv`, falls back to `~/.claude/ml-engineer/venv`.
- Created on first use with one-time user approval.
- Empty by default — packages installed on demand when scripts raise `ModuleNotFoundError` (per-install approval).
- Shared across all sessions.

## Workdir

Each task writes to `./newton_workdir/<UTC-timestamp>/` in the user's current directory:

```
newton_workdir/2026-04-29T14-22-09Z/
├── step_1_load.py
├── verify_step_1.py
├── step_2_explore.py
├── verify_step_2.py
├── output_summary.txt
└── charts/
    ├── correlation_matrix.png
    └── target_distribution.png
```

The workdir belongs to the user; the plugin never deletes it.

## Hard constraints (enforced by the skills)

- No `plt.show()`, `input()`, web servers, infinite loops, or destructive file ops in generated code.
- No system Python, no global pip, no `sudo`.
- Code only runs from inside `newton_workdir/<timestamp>/`.
- Charts saved to `charts/` and announced via `print("Chart saved as ...")`.
- No completion claim without fresh verification evidence.
- No fix without root-cause investigation first.

## Inspiration & credit

The skill structure, "Iron Law" framing, trigger-only descriptions, severity-tagged review pattern, and 4-phase debug methodology were inspired by [obra/superpowers](https://github.com/obra/superpowers) — Jesse Vincent's agentic skills framework. We borrowed the patterns most relevant to a data-science workflow (verification discipline, root-cause debugging, end-of-task review, autonomous orchestration via skill descriptions) and skipped the parts that didn't fit (TDD-as-religion, git-worktree ceremony, branch-finishing flow).

The CV-first / metric-first / project-layout / leakage-pattern discipline in `ml-engineer-cv-design`, `ml-engineer-pick-metric`, the Layout B project skeleton in `ml-engineer-write-code`, and the ML-specific failure patterns added to `ml-engineer-verify` follow the methodology from **Abhishek Thakur's *Approaching (Almost) Any Machine Learning Problem*** — specifically the cross-validation, evaluation metrics, arranging-ML-projects, categorical variables, feature selection, hyperparameter optimization, and ensembling chapters. The recipes (Sturge's rule for binning regression targets, GroupKFold for grouped data, AUC≈1 → suspect, target-encoding-only-inside-folds, the `train.py --fold N --model X` idiom, the per-model HPO ranges) are his.

The plugin shape itself follows Anthropic's official [Claude Code plugin specification](https://code.claude.com/docs/en/plugins-reference) and [Skills authoring guide](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf).

## License

MIT
