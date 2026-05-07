---
name: re-write-up
description: Use when re-detect-plateau returns "stop-and-write" or the user interrupts a research-engine session. Produces the final report (what was tried, what worked, what didn't, why, the new champion, recommended next steps). Only fires from inside the research-engine agent. Do NOT invoke from ml-engineer, cv-engineer, nlp-engineer, llm-engineer, or from a user request.
license: MIT
metadata:
  source: ml-engineer
  version: 0.3.0
---

# Write Up

Produce the engine's final report. The report is the user's takeaway — they read this once, may never read the workdir directly. It must be self-contained, accurate, and honest about negatives.

## When to invoke

- `re-detect-plateau` returned `stop-and-write` (or status.json `next_action: stop_and_write`).
- The user interrupted the engine and the in-flight iteration completed cleanly.
- The user-stated target was hit AND the user chose to stop (Step 3 of `re-detect-plateau`).

## When NOT to invoke

- The engine is still running (status.json `state: running` and no stop signal).
- A user one-shot question — engines don't run for one-shot questions.

## Process

### Step 1 — Read inputs

- `<workdir>/research_engine/dossier.md` — original framing (and any re-framings).
- `<workdir>/research_engine/narrative.md` — full narrative.
- `<workdir>/research_engine/leaderboard.jsonl` — every iteration's metric + cost.
- `<workdir>/research_engine/hypotheses.jsonl` and `hypotheses_archive.jsonl` — full hypothesis history.
- `<workdir>/research_engine/status.json` — final champion, total spend, zoom-out count.
- `<workdir>/research_engine/champion/results.json` — champion details (follow the `champion/` symlink).

### Step 2 — Aggregate facts

Compute (do NOT estimate; read from the files):

- `total_iterations` = count of leaderboard.jsonl records.
- `total_spend_usd` = sum of `cost_usd` across leaderboard records (skip null/missing).
- `started_ts` = earliest `started` timestamp across leaderboard records.
- `stopped_ts` = current UTC timestamp.
- `champion_metric` = from status.json.
- `champion_iter` = from status.json.
- `baseline_metric` = from dossier (if "unknown — engine will establish", use the metric from the first scored iteration).
- `target_status` = "target hit" if dossier had a target and champion meets it; else "target not hit; gap = <X>"; else "no target stated".
- `top_5_wins` = leaderboard records where `champion_after: true` (chronological).
- `top_5_losses` = leaderboard records where `status: failed` or `vs_champion < 0` (worst first).
- `reframings` = count and details from narrative.md `## Re-framed at iter <N>` sections.
- `surprises` = adversarial-source hypotheses (`hypotheses.jsonl` records with `source: adversarial`) that ended up in `top_5_wins`. The engine's anti-local-optimum reflex working.

### Step 3 — Build the report

Write `<workdir>/research_engine/REPORT.md` with the following sections, in order. Every section is mandatory; if a section has no content, write `(none)` explicitly — never silently omit.

```
# Research Engine Report — <problem name>

**Started:** <started_ts>   **Stopped:** <stopped_ts>   **Total iterations:** <n>   **Total spend:** $<value>

## Outcome

**Champion:** <metric value> (iter <NNN>) — <one-line description of the winning approach>
**vs. Baseline:** <delta, sign, %>
**vs. Target:** <"target hit" | "target not hit; gap = <X>" | "no target stated">

## What worked

- <one-line claim> — iter <NNN>: <metric> ± vs prior champion
- <one-line claim> — iter <NNN>: <metric> ± vs prior champion
- ... (top 3-5 wins, chronological — the path through hypothesis space, not just the final winner)

## What didn't (and why)

- <one-line claim> — iter <NNN>: tested, ruled out because <one-line reason from narrative>
- <one-line claim> — iter <NNN>: tested, ruled out because <one-line reason>
- ... (top 3-5 losses minimum — failure modes are knowledge)

## Surprises

- <one-line claim> — <one-line surprise: e.g., "the adversarial wild-card from iter NNN ended up in the top 5">
- ... (or `(none)` if no adversarial wild-card or cross-domain analogy made the top wins)

## Re-framings

- <axis> at iter <N> — <one-line why> — <one-line outcome>
- ... (or `(none)` if no zoom-outs occurred)

## Recommended next steps

- **If the user wants to continue optimizing**: <one-line direction, including what zoom-out axis or what new data would help>
- **If the user wants to ship the champion**: <one-line on what to monitor, what's brittle, what's known to break under load>

## Honest assessment

<one-paragraph assessment that addresses every one of these:
 - Did the engine actually optimize, or did it tunnel? (Look at themes_seen across the run; few themes = tunneling.)
 - Are the claims in the narrative well-supported, or are some only single-source / one-iteration?
 - Would a human researcher disagree with the conclusion? Where?
 - What evidence is missing that, if collected, might change the champion?>
```

### Step 4 — Update status.json

Write:
- `state: stopped`
- `last_event_kind: stopped`
- `last_event: <ISO-8601 UTC ts>`
- `next_action: null`

### Step 5 — Surface the report path to the engine

Return the absolute path to `REPORT.md` so the engine can print it to the user.

## Output format

Return to the engine:

```
REPORT WRITTEN: <abs path to REPORT.md>
CHAMPION: <metric value> (iter <NNN>)
TOTAL ITERATIONS: <n>
TOTAL SPEND: $<value>
NEXT: surface report to user, exit engine.
```

## Verification gates

Before returning to the engine, confirm:

- [ ] `REPORT.md` has all 8 mandatory sections (top-level header, Outcome, What worked, What didn't, Surprises, Re-framings, Recommended next steps, Honest assessment) — even if some contain `(none)`.
- [ ] The "What didn't" section has at least 3 entries (or fewer if total iterations < 3, in which case it has all of them).
- [ ] The "Honest assessment" paragraph addresses all four prompts (tunnel? single-source? human disagreement? missing evidence?).
- [ ] Every claim in the report cites an iteration number — no unsourced claims.
- [ ] The "Recommended next steps" section has BOTH a continuation option AND a shipping option (not just one).
- [ ] `status.json` was updated with `state: stopped`, `next_action: null`.
- [ ] All numbers in the report (iterations, spend, champion metric) match the source files exactly — no rounding errors that lose precision.

If any gate fails, do not return to the engine — fix and re-verify.

## Hard constraints

- The "Honest assessment" section is non-optional. The engine's discipline is honesty, including about its own limitations.
- The "What didn't" section must include at least the top 3 ruled-out claims (or all of them if fewer were ruled out), with iteration numbers and reasons. A report that only celebrates wins is a bug.
- Cite iteration numbers for every claim. Unsupported claims in the report are rejected.
- "Recommended next steps" must include both continuation and shipping options — the user picks based on context.
- Numbers in the report come from the source files, not estimates. The engine has the data; use it.
