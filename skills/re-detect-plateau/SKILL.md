---
name: re-detect-plateau
description: Use after each re-update-narrative call in a research-engine session. Reads narrative deltas (NOT the metric) over the last K iterations. Returns continue / continue-but-diversify / zoom-out / stop-and-write. Only fires from inside the research-engine agent. Do NOT invoke from ml-engineer, cv-engineer, nlp-engineer, llm-engineer, or from a user request.
license: MIT
metadata:
  source: ml-engineer
  version: 0.3.0
---

# Detect Plateau

Decide whether the engine should continue, zoom out, or stop. The signal is **narrative plateau** — no new ruled-out claims and no new suspected claims for K iterations — not metric plateau alone. Metric plateau without narrative plateau means stuck-in-local-well; the engine has more to try.

## When to invoke

- Immediately after every `re-update-narrative` call.

## When NOT to invoke

- The current iteration is still in progress.
- A user one-shot question — engines don't run for one-shot questions.

## Process

### Step 1 — Read inputs

- `<workdir>/research_engine/iterations/<NNN>/narrative_delta.md` for the last K iterations (where K is from Step 2). Iteration directories are zero-padded `001`, `002`, ... — sort numerically.
- `<workdir>/research_engine/leaderboard.jsonl` — for metric-plateau check and average iteration cost.
- `<workdir>/research_engine/dossier.md` — for stop criteria (target hit?) and metric direction.
- `<workdir>/research_engine/status.json` — for current champion and zoom-out history.

### Step 2 — Set adaptive K

K is the number of recent iterations to inspect. Compute the average `cost_usd` over the last 5 leaderboard entries (or all entries if fewer than 5). Skip entries where `cost_usd` is null or missing — average over the remainder. If all 5 entries have null cost, treat the average as 0 (cheapest tier). Adaptive rule:

| average iteration cost | K |
|---|---|
| ≤ $0.50 | 5 |
| > $0.50 and ≤ $2.50 | 4 |
| > $2.50 | 3 |

Cheap experiments warrant longer plateau windows because each one is low-information; expensive experiments warrant shorter ones because each one carries more signal.

If fewer than K iterations have completed, return `continue` immediately. Engine plateau detection requires at least K iterations of history.

### Step 3 — Check for target hit

First, read `status.json.target_hit_resolved` — if absent, treat as `false`. If `true`, the user already answered the target-hit question on a prior iteration; skip this step and proceed to Step 4. This skip is non-optional — the engine asks at most once per session.

Otherwise, read the `Target` field in `dossier.md`. If it has a numeric value AND the current `champion_metric` from `status.json` meets or exceeds it (using the metric direction from the dossier), ask the user once (this is the one allowed proactive question after engine start):

> "Target metric reached (champion: <value>, target: <value>). Continue past the target, or stop?"

Wait for response. Then, regardless of the answer, write `target_hit_resolved: true` to `status.json` so subsequent plateau checks skip this step. If the user said stop: return `stop-and-write`. If continue: proceed to Step 4.

(The persistence is on its own dedicated field, NOT on `last_event_kind`, because Step 7 rewrites `last_event_kind` on every plateau check and would clobber any marker stored there.)

### Step 4 — Compute narrative-plateau signal

Look at the last K `narrative_delta.md` files. Sum:

- `total_added_ruled_out` = sum of "Added to Ruled out" entries across the K deltas (count entries, ignoring `(none)` placeholders).
- `total_added_suspected` = sum of "Added to Currently suspected" entries (same dedup of `(none)`).
- `total_archived` = sum of "Hypotheses archived" entries (ignoring `(none)`).
- `themes_seen` = distinct `theme` values among the K iterations' hypotheses (read from each iteration's `hypothesis.json`).

**Narrative plateau** if all of the following hold:
- `total_added_ruled_out + total_added_suspected ≤ 1`
- `total_archived ≤ 1`
- `themes_seen ≤ 2` (the engine has been tunneling)

### Step 5 — Compute metric-plateau signal

Look at the last K iterations' `metric_value` from leaderboard.jsonl. **Metric plateau** if:

- `(max - min) / abs(champion_metric) < 0.005` (less than 0.5% variation across K iterations).

If `champion_metric` is 0 or null, treat the denominator as 1 (avoid divide-by-zero) — the absolute range is the signal.

Skip iterations whose `metric_value` is `null` (failed runs) when computing min/max but COUNT them in K (a failed run is a real iteration).

### Step 6 — Decide

| Narrative plateau | Metric plateau | Decision |
|---|---|---|
| No | No | `continue` (engine is making progress) |
| No | Yes | `continue` (we're learning even if metric is flat — the narrative will eventually break the plateau) |
| Yes | No | `continue-but-diversify` (metric is moving; narrative isn't — we're surfing one theme. The engine signals `re-generate-hypotheses` to refresh the live list with extra adversarial weight.) |
| Yes | Yes | `zoom-out` (stuck in local well; we need a re-frame) |

After the *first* zoom-out for a given problem, the next plateau (Yes,Yes) signal must NOT trigger another zoom-out within 2*K iterations of the prior one — escalate to `stop-and-write` instead. Two consecutive plateaus are the engine's "I've genuinely run out of moves" signal.

Persistence: `re-zoom-out` writes `zoom_out_count` (incremented) and `last_zoom_out_iter` to `status.json` when it runs. This skill READS those fields:

- If `zoom_out_count` is absent or 0 → no prior zoom-out; on (Yes,Yes), return `zoom-out`.
- If `zoom_out_count >= 1` AND `(current_iter - last_zoom_out_iter) < 2*K` → return `stop-and-write` instead of `zoom-out`.
- Otherwise (≥ 2*K iterations since last zoom-out) → return `zoom-out` (the engine has had enough new evidence to warrant another reframe).

This skill never writes `zoom_out_count` or `last_zoom_out_iter`. Those are owned by `re-zoom-out`.

### Step 7 — Update status

Write to `status.json`:

- `last_event_kind: plateau_check`
- `last_event: <ISO-8601 UTC ts>`
- `next_action`: translate the decision to the next skill name (the schema's `next_action` enum names skills, not decisions):

| Decision (Step 6) | `next_action` written to status.json |
|---|---|
| `continue` | `re_select_next` |
| `continue-but-diversify` | `re_generate_hypotheses` |
| `zoom-out` | `re_zoom_out` |
| `stop-and-write` | `re_write_up` |

This translation makes resume work: on resume, the engine reads `next_action` from `status.json` and dispatches to the named skill (mapping snake_case → kebab-case: `re_select_next` → `re-select-next`).

Also store the decision itself in a separate field `last_plateau_decision` (free-form snake_case: `continue`, `continue_but_diversify`, `zoom_out`, `stop_and_write`) so reports / debugging can trace what was decided. This is a debug field; the engine's resume path uses `next_action`.

## Output format

Return to the engine:

```
PLATEAU CHECK: iter <NNN>, K=<value>
NARRATIVE PLATEAU: <yes/no> (added: <n>, archived: <n>, themes: <n>)
METRIC PLATEAU: <yes/no> (range: <%>)
DECISION: <continue | continue-but-diversify | zoom-out | stop-and-write>
NEXT: <re-select-next | re-generate-hypotheses (diversify mode) | re-zoom-out | re-write-up>
```

## Verification gates

Before returning to the engine, confirm:

- [ ] K was computed from actual leaderboard cost data, not a hardcoded value.
- [ ] At least K iterations were inspected (or `continue` was returned for insufficient history).
- [ ] If the dossier had a numeric target AND it was reached, the user was asked exactly once and the answer was honored. The user is NOT re-asked on subsequent plateau checks (verify status.json.target_hit_resolved).
- [ ] The narrative-plateau signal counted only non-`(none)` entries.
- [ ] The metric-plateau signal handled null metric values correctly (skipped when computing range, counted in K).
- [ ] The decision matches the table in Step 6 exactly. Do NOT invent a new decision category.
- [ ] If two consecutive plateaus were detected, the decision is `stop-and-write`, not `zoom-out`.
- [ ] `status.json` was updated with `next_action` set to the translated skill name (one of `re_select_next` / `re_generate_hypotheses` / `re_zoom_out` / `re_write_up`), AND `last_plateau_decision` was set to the original decision string.

If any gate fails, do not return to the engine — fix and re-verify.

## Hard constraints

- The decision table is the contract. Do not invent new decision categories.
- Plateau is computed from the narrative, NOT the metric alone. A metric-only plateau detector is a bug — it cannot escape local optima.
- The "two consecutive plateaus → stop-and-write" rule prevents infinite zoom-out loops.
- Target-hit asks the user exactly once per session. Subsequent plateau checks after a target-pass-through do not re-ask.
- `status.json.next_action` enum values are snake_case skill names from the schema's enum. The plateau decision strings (`continue`, `zoom_out`, etc.) live in `last_plateau_decision`. Do not put decision strings in `next_action` — they are not in the schema's enum and will break resume.
