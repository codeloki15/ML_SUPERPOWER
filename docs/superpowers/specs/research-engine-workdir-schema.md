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

> Examples are illustrative; values across examples are not meant to tell a single consistent story.

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

Note: there are TWO distinct `status` enums in this schema — one for hypothesis records (here) and one for leaderboard records (below) — and values from one must NOT be written to the other.

`rank`: integer, 1 = highest priority for selection. Re-ranked by `re-update-narrative` after each iteration.

`expected_gain`, `expected_cost` ∈ `{low, med, high}`.

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

`status` ∈ `{scored, failed, debug_exhausted}`.

`compute`: free-form string identifying the compute environment (e.g. `local-cpu`, `local-mps`, `local-cuda`, `modal`, `runpod-rtx4090`, `vast-h100`). Set by `dl-detect-env` / `dl-remote-execute`; engine treats this as opaque.

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
  "user_paused": false,
  "target_hit_resolved": false,
  "zoom_out_count": 0,
  "last_zoom_out_iter": null
}
```

`state` ∈ `{initializing, running, awaiting_user, paused, stopped}`.

`next_action` ∈ `{re_frame_problem, re_mine_literature, re_generate_hypotheses, re_select_next, dispatch_to_subagent, re_update_narrative, re_detect_plateau, re_zoom_out, re_write_up, awaiting_user_response, null}` — `null` means engine is stopped.

`last_event_kind` ∈ `{engine_started, framing_complete, literature_pass_complete, hypotheses_generated, experiment_selected, iteration_complete, plateau_check, zoom_out_complete, awaiting_user_response, stopped}`.

`target_hit_resolved` (bool) — `true` once the user has answered the one-shot "target reached, continue or stop?" question. Owned by `re-detect-plateau`; never reset within a session.

`zoom_out_count` (int) — number of zoom-outs performed in this session. Owned by `re-zoom-out` (incremented when it runs); read by `re-detect-plateau` to decide whether to escalate to `stop_and_write` instead of repeating zoom-out.

`last_zoom_out_iter` (int or null) — `current_iter` at the moment of the most recent zoom-out. Owned by `re-zoom-out`; read by `re-detect-plateau` to compute "iterations since last zoom-out."

`reframe_axis` (used by `re-zoom-out` when re-invoking `re-frame-problem`, not stored in `status.json`) ∈ `{metric, unit_of_analysis, decomposition, data_slice}` — the axis along which the engine re-frames the problem on a zoom-out. See `re-zoom-out` Step 2 for the deterministic selection rule.

`last_plateau_decision` (string, optional debug field) — written by `re-detect-plateau` alongside `next_action`. Records the original decision string for traceability: ∈ `{continue, continue_but_diversify, zoom_out, stop_and_write}`.

### `iterations/<NNN>/`

One directory per executed experiment. Filled by the domain sub-agent's existing skills (the engine just provides the directory path).

- `hypothesis.json` — exact record from `hypotheses.jsonl` at the time the experiment was selected.
- `plan.md` — output of `ml-engineer-plan`.
- `step_*.py` / `verify_*.py` — outputs of `ml-engineer-write-code` / `ml-engineer-verify`.
- `results.json` — final metric, written by `ml-engineer-verify`. Required fields: `metric_name`, `metric_value`, `verified` (bool).
- `narrative_delta.md` — markdown receipt for this iteration. Written by `re-update-narrative`. Required sections: `## Added to "Ruled out"`, `## Added to "Currently suspected"`, `## Removed from "Open questions"`, `## Hypotheses archived`, `## Champion changed?`. Each section may contain `(none)` if empty.
- `selection_note.md` (optional) — markdown receipt written by `re-select-next` ONLY when a random tie-break occurred. Records the tied candidates and the random outcome. `re-update-narrative` folds this content into `narrative_delta.md` under a `## Selection note` section.

### `reading/<TS>.md`

`<TS>` format: ISO-8601 UTC with `:` replaced by `-` and seconds dropped, i.e. `YYYY-MM-DDTHH-MMZ` — matching the example `2026-05-07T14-22Z.md`.

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
