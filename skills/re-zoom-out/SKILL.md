---
name: re-zoom-out
description: Use when re-detect-plateau returns "zoom-out" during a research-engine session. Forces a perspective shift — different metric, different unit of analysis, or different decomposition. Appends to (does not overwrite) the narrative. Owns zoom_out_count and last_zoom_out_iter in status.json. Only fires from inside the research-engine agent. Do NOT invoke from ml-engineer, cv-engineer, nlp-engineer, llm-engineer, or from a user request.
license: MIT
metadata:
  source: ml-engineer
  version: 0.3.0
---

# Zoom Out

Escape the local optimum. The engine has run K iterations without producing new claims AND without metric movement. The current framing is exhausted; a different framing might unlock new directions.

The discipline: **change the framing, do not delete the narrative.** The user might want to come back to the original framing later; the engine should not destroy work.

## When to invoke

- Only when `re-detect-plateau` returned `zoom-out` (kebab-case in the output, `zoom_out` in `status.json.next_action`).

## When NOT to invoke

- Any other plateau-check outcome (`continue`, `continue-but-diversify`, `stop-and-write`).
- A user one-shot question — engines don't run for one-shot questions.

## Process

### Step 1 — Read inputs

- `<workdir>/research_engine/dossier.md` — current framing.
- `<workdir>/research_engine/narrative.md` — current claim set.
- `<workdir>/research_engine/leaderboard.jsonl` — what's been tried.
- `<workdir>/research_engine/hypotheses_archive.jsonl` — what's been archived (and may be revivable under a new framing).
- `<workdir>/research_engine/status.json` — current state, current_iter, prior zoom_out_count.

### Step 2 — Identify the framing axis to shift

Pick exactly one of the following four axes for this zoom-out. Pick whichever is most likely to unlock signal given the narrative:

1. **metric** — the current primary metric may be the wrong target. Shift to a sibling metric (e.g., RMSE → MAE, accuracy → F1, BLEU → BERTScore) and re-baseline. Does the leaderboard rank the same? If not, the metric was the problem.

2. **unit_of_analysis** — the current unit may be wrong. Examples: per-row → per-group; per-token → per-sequence; per-image → per-patient. Shift the unit and recompute the champion.

3. **decomposition** — the problem may be a hidden composite. Examples: a classification task may be better as a 2-stage detect-then-classify; a regression may be better as classification of bins + within-bin regression. Force a 2-stage decomposition and re-baseline the easier sub-task first.

4. **data_slice** — the problem may be heterogeneous. Examples: stratify by a feature and check whether the champion is uniformly good or only good on a subset. If the latter, the new framing is "solve the bad-subset specifically."

The chosen axis name uses snake_case to match `status.json` enum convention.

### Step 3 — Re-invoke `re-frame-problem` with the new axis

Call `re-frame-problem` passing two arguments (per its `When to invoke` contract):
- `reframe_axis` ∈ `{metric, unit_of_analysis, decomposition, data_slice}` — the axis chosen in Step 2.
- `reframe_reason` — one-line explanation derived from the narrative (e.g., "metric plateau + theme tunneling on regularization for 5 iterations; switching to MAE to test whether RMSE was the wrong target").

`re-frame-problem` will:
- Append a `## Re-framed at iter <N>` section to `dossier.md` (NOT overwrite Step 2 fields wholesale; only the relevant ones).
- Append a `## Re-framed at iter <N>` section to `narrative.md` (NOT overwrite).
- Preserve all "Already known to work / fail" content.

### Step 4 — Categorize archived hypotheses under the new framing

Read `hypotheses_archive.jsonl`. Some may be *revivable* under the new framing. For each archived hypothesis, decide:

- **Revive** — its `concrete_change` is still meaningful under the new framing → write a new live record in `hypotheses.jsonl` with a fresh `id`, `parent_id` set to the archived id, `source: mutation` (the revival is a mutation under the new context), and an annotation in the per-iteration log: "previously archived because <X>; revived because <Y>".
- **Stay archived** — its `concrete_change` is irrelevant under the new framing → no action; the record stays in `hypotheses_archive.jsonl`.

### Step 5 — Append to narrative

Append the following block to `<workdir>/research_engine/narrative.md` (do NOT overwrite existing sections):

```
## Re-framed at iter <N>
**Axis:** <metric | unit_of_analysis | decomposition | data_slice>
**Why:** <one-line reason from the plateau analysis>
**What changed:**
- <one-line bullet>
- <one-line bullet>
**Existing claims still valid:** <list of "Currently suspected" entries that survive>
**Existing claims now invalid:** <list of "Ruled out" or "Currently suspected" entries that no longer make sense>
**Revived hypotheses:** <list of revived ids>
```

### Step 6 — Update status.json

This skill owns `zoom_out_count` and `last_zoom_out_iter` in `status.json`. Atomically update:

- `zoom_out_count: <prior + 1>` (read prior; default 0 if absent).
- `last_zoom_out_iter: <current_iter>` (current iteration number from `status.json`).
- `last_event_kind: zoom_out_complete`
- `last_event: <ISO-8601 UTC ts>`
- `next_action: re_generate_hypotheses` — the engine's next step is initial-seed-mode hypothesis generation under the new framing.

### Step 7 — Trigger `re-generate-hypotheses` in initial-seed mode

After this skill returns, the engine will call `re-generate-hypotheses`. The new framing is effectively a new engine start in disguise — but with the existing narrative as scaffolding, not a blank slate. Initial-seed quotas apply (~20 candidates, 5+ literature, 3+ cross-domain, 2+ adversarial). Mutation/failure-mining quotas are NOT exempt because revived archived hypotheses (Step 4) supply the seed for both.

## Output format

Return to the engine:

```
ZOOMED OUT: axis = <metric | unit_of_analysis | decomposition | data_slice>
NEW FRAMING: <one-line summary>
HYPOTHESES REVIVED: <n>
CLAIMS PRESERVED: <n>
CLAIMS INVALIDATED: <n>
ZOOM_OUT_COUNT: <new value>
NEXT: re-generate-hypotheses (initial-seed mode under new framing)
```

## Verification gates

Before returning to the engine, confirm:

- [ ] Exactly ONE axis was picked (multi-axis re-framings are rejected; if multiple axes look promising, pick the highest-leverage one for THIS zoom-out — the next zoom-out can pick another).
- [ ] `dossier.md` has a new `## Re-framed at iter <N>` section appended (not overwriting Step 2 fields wholesale).
- [ ] `narrative.md` has a new `## Re-framed at iter <N>` section appended; the original `## Ruled out`, `## Currently suspected`, `## Open questions`, `## Per-iteration log` sections still contain their prior entries unchanged.
- [ ] `hypotheses_archive.jsonl` was NOT modified — revivals create new records in `hypotheses.jsonl`, the archive is read-only here.
- [ ] `status.json.zoom_out_count` was incremented by exactly 1.
- [ ] `status.json.last_zoom_out_iter` equals the `current_iter` at invocation.
- [ ] `status.json.next_action` is `re_generate_hypotheses`.
- [ ] Every revived hypothesis has `parent_id` set to the archived original's id and `source: mutation`.

If any gate fails, do not return to the engine — fix and re-verify.

## Hard constraints

- Pick exactly ONE axis per zoom-out. Multi-axis re-framings are rejected — each axis is its own zoom-out.
- Never delete `narrative.md` content. Append only.
- Never delete entries from `hypotheses_archive.jsonl`. Reviving creates a new record in `hypotheses.jsonl` with `parent_id`.
- The original dossier is preserved. The re-framing appends a section, does not overwrite.
- This skill is the SOLE writer of `zoom_out_count` and `last_zoom_out_iter` in `status.json`. No other skill modifies these fields.
