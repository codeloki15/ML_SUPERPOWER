---
name: re-frame-problem
description: Use at the start of a research-engine session to build the problem dossier (metric, baseline, data shape, prior knowledge, stop criteria). Re-invoked by re-zoom-out when the engine needs to escape a local optimum. Only fires from inside the research-engine agent. Do NOT invoke from ml-engineer, cv-engineer, nlp-engineer, llm-engineer, or from a user request.
license: MIT
metadata:
  source: ml-engineer
  version: 0.3.0
---

# Frame Problem

Build the **problem dossier** that anchors the research engine. The dossier is read by every other `re-*` skill — it determines what counts as a hypothesis, what counts as a win, and when the engine stops.

This skill captures what the user told the engine + what can be inferred from the data + what's already known to work or fail. It does not invent claims; it records them with sources.

## When to invoke

- First action after the `research-engine` agent is dispatched, before any other `re-*` skill.
- Re-invoked by `re-zoom-out` with an explicit instruction to change the framing (different metric, different unit of analysis, different decomposition).

## When NOT to invoke

- The engine is already running and the dossier exists — use the dossier; do not rebuild it.
- A user one-shot question — engines don't run for one-shot questions.

## Process

### Step 1 — Locate or create the workdir

The `research-engine` agent passes you the absolute path to `<workdir>/research_engine/`. If the directory does not exist, create it. If `dossier.md` already exists and you were not invoked by `re-zoom-out`, refuse and instruct the engine to read the existing dossier.

### Step 2 — Extract from the user's request

Identify:
- **Problem statement** — one paragraph. What is the user trying to optimize / discover / beat?
- **Primary metric** — name and direction (higher-is-better or lower-is-better). If the user did not name one, infer from the task type and verify with one targeted question to the user (e.g., "Should I optimize for AUC or F1?"). This is one of the few user questions the engine is allowed.
- **Baseline** — current value the user is trying to beat. Numeric, with the same metric. May be `unknown — engine will establish` if the user has no baseline.
- **Target** — value the user wants to reach. May be `none — push frontier as far as possible`.

### Step 3 — Probe the data shape

If the user attached data, run a Layout-A EDA probe through `ml-engineer-write-code` → `ml-engineer-execute` to get rows / columns / modality / size on disk / dtype distribution / class balance / known quirks. The probe is mandatory; do not write the dossier without it. The probe's output goes into the `## Data shape` section as one paragraph (not raw output).

If the user did not attach data, mark the section `pending — first iteration will probe the user-provided source`.

### Step 4 — Capture prior knowledge

Two sections, each pulled from the user's message + (if non-empty) `dl-prior-art` if invoked earlier this session:

- **Already known to work** — claims with sources.
- **Already known to fail** — claims with sources.

If both are empty, write `(none stated yet — narrative will accumulate as the engine runs)`.

### Step 5 — Capture stop criteria

From the user's message or sensible defaults:
- Target hit: yes / no (yes if Step 2 produced a target).
- Time budget: hours, or `unbounded` (default `unbounded`).
- Cost ceiling: dollars per session, default `5`.
- Other: free-form.

### Step 6 — Write `dossier.md`

Use the schema in `docs/superpowers/specs/research-engine-workdir-schema.md` exactly. Required sections, in order: Problem statement / Metric / Data shape / Already known to work / Already known to fail / Stop criteria.

### Step 7 — Initialize `status.json` and the narrative skeleton

If invoked at engine start, also create:
- `status.json` with `state: initializing`, `current_iter: 0`, `champion_iter: 0`, `champion_metric: null`, `spend_so_far_usd: 0`, `cost_ceiling_usd` from Step 5.
- `narrative.md` with empty `## Ruled out`, `## Currently suspected`, `## Open questions`, `## Per-iteration log` sections and the header.

If invoked by `re-zoom-out`, do NOT overwrite `narrative.md` — append a `## Re-framed at iter <N>` section to the existing narrative explaining what changed in the framing.

## Output format

Return to the engine:

```
DOSSIER WRITTEN: <abs path to dossier.md>
PRIMARY METRIC: <name> (<direction>)
BASELINE: <value or 'unknown'>
TARGET: <value or 'none'>
DATA SHAPE: <one-line summary>
COST CEILING: $<value>
NEXT: re-mine-literature
```

## Hard constraints

- Never invent a baseline. If unknown, write `unknown — engine will establish`.
- Never invent claims for "already known to work / fail". If empty, say so.
- The data-shape probe is mandatory if data was provided. No probe = no dossier.
- One — at most one — clarifying question to the user is allowed (and only for the metric, only if undecidable from the request). All other gaps go in as `pending` or `none stated`.
