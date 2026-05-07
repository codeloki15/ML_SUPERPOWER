# Research Engine Workdir Schema

This file defines the layout of `<workdir>/research_engine/`, where `<workdir>` is `./newton_workdir/<UTC-timestamp>/`. Every `re-*` skill reads/writes paths defined here.

## Layout

```
<workdir>/research_engine/
├── dossier.md
├── narrative.md
├── hypotheses.jsonl
├── hypotheses_archive.jsonl
├── leaderboard.jsonl
├── status.json
├── champion/                   # symlink to iterations/<NNN>/ of the current best run
├── iterations/
│   ├── 001/
│   │   ├── hypothesis.json
│   │   ├── plan.md
│   │   ├── step_*.py
│   │   ├── verify_*.py
│   │   ├── results.json
│   │   └── narrative_delta.md
│   ├── 002/
│   └── ...
└── reading/
    ├── 2026-05-07T14-22Z.md
    └── ...
```

## File specs

### `dossier.md`

Markdown. Sections (all required):

```markdown
# Problem dossier
**Started:** <ISO-8601 UTC>   **Last updated:** <ISO-8601 UTC>

## Problem statement
<one paragraph>

## Metric
**Primary:** <metric name> (<higher-is-better | lower-is-better>)
**Baseline:** <numeric value, or `unknown — engine will establish` if no baseline yet>
**Target:** <numeric value, or `none — push frontier as far as possible`>

## Data shape
<one paragraph: rows / columns / modality / size on disk / known quirks>

## Already known to work
- <claim> (source: <user | literature | prior run>)

## Already known to fail
- <claim> (source: <user | literature | prior run>)

## Stop criteria (user-stated)
- Target hit: <yes | no>
- Time budget: <hours, or `unbounded`>
- Cost ceiling: <dollars per session, default 5>
- Other: <free-form>
```

### `narrative.md`

Markdown. Sections:

```markdown
# Narrative — <problem name>
**Started:** <ts>   **Last updated:** <ts>   **Champion:** <metric value> (iter <n>)

## Ruled out
- <claim> (iter <n>: <one-line reason>)

## Currently suspected
- <claim> (iter <n>: <one-line reason>) [confidence: low|med|high]

## Open questions
- <question> (raised iter <n>: <one-line reason>)

## Per-iteration log
### Iter 001 — <hypothesis one-line>
**Result:** <metric, ± vs champion>
**Implies:** <one-line takeaway>
**Narrative delta:** added [...], removed [...], promoted [...] from suspected to ruled-out.
```

### `hypotheses.jsonl`

JSONL — one record per live hypothesis. Append-only; updates write a new record with the same `id` and a higher `version`.

```json
{
  "id": "h_001",
  "version": 1,
  "created_iter": 0,
  "updated_iter": 0,
  "status": "live",
  "theme": "regularization",
  "one_line": "Add weight decay 1e-2 — winner playbook says it helps on this data shape.",
  "concrete_change": "Set weight_decay=1e-2 in optimizer; keep all else fixed.",
  "source": "literature",
  "parent_id": null,
  "expected_gain": "med",
  "expected_cost": "low",
  "rank": 1
}
```

`source` ∈ `{literature, mutation, failure_mining, cross_domain, adversarial}`. `status` ∈ `{live, running, scored, archived}`. `parent_id` is non-null for `mutation` source.

### `hypotheses_archive.jsonl`

Same schema as `hypotheses.jsonl`. Records moved here when killed. Adds `archive_reason: <string>` and `archived_iter: <int>`.

### `leaderboard.jsonl`

JSONL — one record per executed experiment.

```json
{
  "iter": 1,
  "hypothesis_id": "h_001",
  "metric_name": "rmse",
  "metric_value": 3.41,
  "vs_champion": -0.03,
  "champion_after": true,
  "cost_usd": 0.12,
  "compute": "local-mps",
  "started": "2026-05-08T14:22:09Z",
  "ended": "2026-05-08T14:31:47Z",
  "status": "scored"
}
```

`status` ∈ `{scored, failed, debug-exhausted}`.

### `status.json`

Single JSON object — current engine state.

```json
{
  "engine_version": "0.3.0",
  "state": "running",
  "current_iter": 7,
  "champion_iter": 5,
  "champion_metric": 3.31,
  "last_event": "2026-05-08T14:31:47Z",
  "last_event_kind": "iteration_complete",
  "next_action": "select_next",
  "spend_so_far_usd": 0.78,
  "cost_ceiling_usd": 5.0,
  "user_paused": false
}
```

`state` ∈ `{initializing, running, awaiting_user, paused, stopped}`.

### `iterations/<NNN>/`

One directory per executed experiment. Filled by the domain sub-agent's existing skills (the engine just provides the directory path).

- `hypothesis.json` — exact record from `hypotheses.jsonl` at the time the experiment was selected.
- `plan.md` — output of `ml-engineer-plan`.
- `step_*.py` / `verify_*.py` — outputs of `ml-engineer-write-code` / `ml-engineer-verify`.
- `results.json` — final metric, written by `ml-engineer-verify`. Required fields: `metric_name`, `metric_value`, `verified` (bool).
- `narrative_delta.md` — what this iteration added/removed from the narrative. Written by `re-update-narrative`.

### `reading/<TS>.md`

One file per `re-mine-literature` pass. Markdown. Required sections:

```markdown
# Literature pass — <ISO-8601 UTC>
**Triggered by:** <narrative section that prompted the search>

## Queries used
- <query 1>
- <query 2>

## Sources read
- <URL or citation> — <one-line takeaway>

## Claims extracted
- <claim> → narrative section: <ruled-out | suspected | open>
```
