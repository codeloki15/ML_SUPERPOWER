---
name: re-update-narrative
description: Use after each iteration in a research-engine session. Reads the iteration's results.json, rewrites narrative.md (adding/removing/promoting claims), re-ranks live hypotheses, re-baselines if a new champion. The forced "ruled out" / "promoted" fields prevent score-only updates. Only fires from inside the research-engine agent. Do NOT invoke from ml-engineer, cv-engineer, nlp-engineer, llm-engineer, or from a user request.
license: MIT
metadata:
  source: ml-engineer
  version: 0.3.0
---

# Update Narrative

Convert the iteration's outcome into narrative deltas. The narrative is the engine's working memory — every other `re-*` skill reads from it. A score-only update (just the leaderboard, nothing in the narrative) is a bug, because it means the engine learned nothing.

## When to invoke

- Immediately after a sub-agent's transactional loop returns for an iteration. Always. No exceptions.

## When NOT to invoke

- A user one-shot question — engines don't run for one-shot questions.
- The iteration is still in progress (sub-agent has not returned).

## Process

### Step 1 — Read inputs

- `<workdir>/research_engine/iterations/<NNN>/results.json` — the iteration's metric and verification status.
- `<workdir>/research_engine/iterations/<NNN>/hypothesis.json` — what was being tested.
- `<workdir>/research_engine/iterations/<NNN>/selection_note.md` (if present) — random tie-break record from `re-select-next`.
- `<workdir>/research_engine/narrative.md` — the current narrative.
- `<workdir>/research_engine/leaderboard.jsonl` — the running leaderboard.
- `<workdir>/research_engine/status.json` — current champion.
- `<workdir>/research_engine/hypotheses.jsonl` — for re-ranking.

If `results.json` does not exist or `verified: false`, treat the iteration as a *failed* run; the metric is `null` and the result type is `debug-exhausted` or `failed`. Failed runs are still informative (a failure mode is a claim).

### Step 2 — Compute the four forced fields

This step is the discipline. You must produce *every one* of these four; if any is empty, write `(none)` explicitly. An update without the forced structure is rejected.

1. **What was tested** — restate the hypothesis in one sentence.
2. **What the result was** — metric value, vs. champion (delta, sign), pass/fail vs. baseline.
3. **What is now ruled out** — at least one entry, even if "(none — result was inconclusive)". A ruled-out claim must reference a prior `## Currently suspected` or `## Open questions` entry by its text or be a new claim newly demonstrated false.
4. **What is now newly suspected** — at least one entry. May be "(none — no new suspicion raised)". A newly-suspected claim must come from the iteration's actual evidence, not from speculation.

### Step 3 — Append to per-iteration log

Append to `narrative.md` under `## Per-iteration log`:

```
### Iter <NNN> — <hypothesis one-line>
**Result:** <metric, ± vs champion>
**Implies:** <one-line takeaway>
**Narrative delta:** added [<list>], removed [<list>], promoted [<list>] from suspected to ruled-out.
```

### Step 4 — Update the four narrative sections

In `narrative.md`:

- `## Ruled out` — append entries from forced field #3. Promote any matching `## Currently suspected` entries (move them here, deleting from there).
- `## Currently suspected` — append entries from forced field #4. Increment confidence on existing entries if independently corroborated.
- `## Open questions` — remove any questions resolved by this iteration. Add any new questions raised.
- Update the header `**Last updated:**` and `**Champion:**` lines.

### Step 5 — Append to leaderboard

Append a single record to `leaderboard.jsonl`:

```
{
  "iter": <NNN>,
  "hypothesis_id": "<id>",
  "metric_name": "<name>",
  "metric_value": <number or null>,
  "vs_champion": <delta or null>,
  "champion_after": <bool>,
  "cost_usd": <from sub-agent's actual cost>,
  "compute": "<env id>",
  "started": "<ISO-8601 UTC ts>",
  "ended": "<ISO-8601 UTC ts>",
  "status": "scored | failed | debug_exhausted"
}
```

(Status enum follows the schema doc — `scored`, `failed`, `debug_exhausted` — snake_case.)

### Step 6 — Re-baseline if champion changed

Compare the iteration's metric to the current champion (read `status.json` `champion_metric`). Apply the metric direction from `dossier.md` (higher-is-better or lower-is-better). If beaten:

- Update `status.json`: `champion_iter: <NNN>`, `champion_metric: <new value>`.
- Re-create the `champion/` symlink: `rm -f <workdir>/research_engine/champion && ln -s iterations/<NNN> <workdir>/research_engine/champion`.
- Add a header line to the per-iteration log entry: `**🏆 New champion** — was <prev metric>, now <new metric>.`

### Step 7 — Re-rank live hypotheses

Read `hypotheses.jsonl` (latest version per id, status=live only). For each live hypothesis, recompute `rank` based on the new narrative state:

- Hypotheses whose `concrete_change` is now ruled out by the iteration → archive (status `archived`, append to `hypotheses_archive.jsonl` with `archive_reason: "contradicted by iter <NNN>"`, and write a versioned record to `hypotheses.jsonl` with `status: archived`).
- Hypotheses whose theme matches a newly-suspected claim → bump their `expected_gain` up one tier (low → med, med → high). Write a new versioned record to `hypotheses.jsonl`.
- Hypotheses identical (after dedup-rule normalization: lowercase + strip whitespace + strip punctuation) to the just-run hypothesis → archive with `archive_reason: "tested in iter <NNN>"`.

Write versioned records back to `hypotheses.jsonl` (do not overwrite; append with incremented `version`).

### Step 8 — Write `narrative_delta.md` for the iteration

Write `<workdir>/research_engine/iterations/<NNN>/narrative_delta.md` summarizing exactly what changed. The required sections (per the schema doc):

```
# Narrative delta — iter <NNN>

**Hypothesis tested:** <one-line>
**Result:** <metric ± vs champion>

## Added to "Ruled out"
- ...   (or `(none)`)

## Added to "Currently suspected"
- ...   (or `(none)`)

## Removed from "Open questions"
- ...   (or `(none)`)

## Hypotheses archived
- <id> (<reason>)   (or `(none)`)

## Champion changed?
<yes / no — old <value> → new <value>>
```

If a `selection_note.md` was present (from a random tie-break in `re-select-next`), append its contents at the bottom of `narrative_delta.md` under a new section `## Selection note` so the random tie-break is folded into the durable record.

## Output format

Return to the engine:

```
NARRATIVE UPDATED: iter <NNN>
RESULT: <metric value> (<delta vs champion>)
NEW CLAIMS: ruled-out=<n>, suspected=<n>, open=<n>
HYPOTHESES ARCHIVED: <n>
CHAMPION CHANGED: <yes/no>
NEXT: re-detect-plateau
```

## Verification gates

Before returning to the engine, confirm:

- [ ] Every one of the four forced fields was filled (not omitted). Empty fields use `(none)` explicitly, not blank.
- [ ] `narrative.md` has exactly one new `### Iter <NNN>` entry under `## Per-iteration log`.
- [ ] `narrative.md` retains all four top-level sections (`## Ruled out`, `## Currently suspected`, `## Open questions`, `## Per-iteration log`) — none deleted.
- [ ] `leaderboard.jsonl` has exactly one new record for this iteration with valid `status` ∈ `{scored, failed, debug_exhausted}`.
- [ ] If the metric beat the champion, `status.json` was updated AND the `champion/` symlink points to `iterations/<NNN>`.
- [ ] If a champion change occurred, the per-iteration log entry has the `**🏆 New champion**` line.
- [ ] `narrative_delta.md` exists in `iterations/<NNN>/` with all 5 required sections.
- [ ] No record in `hypotheses.jsonl` was overwritten — every change is a new versioned record.
- [ ] If `selection_note.md` was present, its contents are folded into `narrative_delta.md` under `## Selection note`.

If any gate fails, do not return to the engine — fix and re-verify.

## Hard constraints

- The four forced fields are the contract. Score-only updates are rejected by skill design — every iteration must produce at least one ruled-out OR newly-suspected claim, even if the claim is "the metric did not move because the change was a no-op."
- A failed iteration is still a narrative event. Write it.
- Champion update + symlink + leaderboard append are atomic-ish; if any of the three fails, do not partially update — return an error and let the engine surface it.
- Never delete entries from `hypotheses.jsonl`. Archive by appending to `hypotheses_archive.jsonl` and writing a versioned record with `status: archived`.
- Never delete content from `narrative.md`. Promotions move entries between sections; nothing is silently removed without a corresponding entry in `narrative_delta.md`.
