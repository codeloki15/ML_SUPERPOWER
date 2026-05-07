---
name: re-select-next
description: Use during a research-engine session to pick the next experiment to run. Scores live hypotheses by expected information gain × cost⁻¹; tie-breaks toward diversity. The brain of the engine. Only fires from inside the research-engine agent. Do NOT invoke from ml-engineer, cv-engineer, nlp-engineer, llm-engineer, or from a user request.
license: MIT
metadata:
  source: ml-engineer
  version: 0.3.0
---

# Select Next Experiment

Pick the next experiment to dispatch. The selector reads the live hypothesis list, the narrative, and the recent leaderboard; it returns one hypothesis to run.

The selector's discipline: maximize **expected information gain × cost⁻¹**. *Information gain* means "how much would this result, regardless of sign, change what the narrative now claims?" Negative results that resolve a debate are high-gain. Positive results that confirm an already-strong claim are low-gain.

## When to invoke

- After `re-generate-hypotheses` (initial round and every iteration thereafter).
- On demand if a previously-selected hypothesis became infeasible (e.g., dependency missing, dataset issue) and a replacement is needed.

## When NOT to invoke

- The live list is empty. Run `re-generate-hypotheses` first.
- A user one-shot question — engines don't run for one-shot questions.

## Process

### Step 1 — Read inputs

- `<workdir>/research_engine/hypotheses.jsonl` — live list. Read all records and use the latest version per id (records are append-only and versioned).
- `<workdir>/research_engine/narrative.md` — for current claim weights.
- `<workdir>/research_engine/leaderboard.jsonl` — for what's recently been run (diversity).
- `<workdir>/research_engine/status.json` — for `spend_so_far_usd` and `cost_ceiling_usd`.

Filter the hypothesis list to records whose latest version has `status: live`.

### Step 2 — Score each candidate

For each live hypothesis, compute:

```
score = gain_weight(expected_gain) × novelty_factor / cost_weight(expected_cost)
```

Weights:

| `expected_gain` | `gain_weight` |
|---|---|
| low | 1 |
| med | 3 |
| high | 9 |

| `expected_cost` | `cost_weight` |
|---|---|
| low | 1 |
| med | 3 |
| high | 9 |

`novelty_factor`:
- 1.0 if the hypothesis's `theme` was NOT used in the last 3 leaderboard entries.
- 0.5 if the theme matches one of the last 3 entries.
- 0.25 if the theme matches all 3 of the last 3 entries.

This is the "tie-break toward diversity" rule. Identical-theme runs in a row are penalized so the engine doesn't tunnel.

### Step 3 — Apply the cost ceiling

Read `status.json`. Compute `remaining = cost_ceiling_usd - spend_so_far_usd`. Estimate the dollar cost of each candidate using this map (these are coarse priors; refined per-iteration as actual costs come in from `dl-remote-execute`):

| `expected_cost` | dollar estimate |
|---|---|
| low | $0 |
| med | $0.50 |
| high | $2.50 |

Filter out any candidate whose estimated cost > `remaining`. If all candidates are filtered, return:

```
ALL CANDIDATES EXCEED COST CEILING.
PROJECTED SPEND OVER NEXT ROUND: $<X>.
ASKING USER FOR APPROVAL: re-select-next-cost-question
```

The engine surfaces a single batched approval question to the user. Do NOT silently demote — the user contract is that the engine asks before paying.

### Step 4 — Pick the highest-scored candidate

Tie-breaks (in order):
1. Higher `expected_gain`.
2. Lower `expected_cost`.
3. Older `created_iter` (don't let new candidates starve old ones forever).
4. Random — and write a `<workdir>/research_engine/iterations/<NNN>/selection_note.md` recording the tie-break and the candidates considered. `re-update-narrative` will fold this into `narrative_delta.md` after the iteration completes.

### Step 5 — Mark and emit

- Update the chosen hypothesis's `status` to `running` in `hypotheses.jsonl` (write a new versioned record — append, do not overwrite).
- Create the iteration directory `<workdir>/research_engine/iterations/<NNN>/` (zero-padded, NNN = current_iter+1) and write `iterations/<NNN>/hypothesis.json` containing the full chosen hypothesis record (the same JSON object that was just versioned in `hypotheses.jsonl`). This file is the source of truth for the iteration: `re-update-narrative` and `re-detect-plateau` read it. Without it, downstream skills cannot complete.
- Update `status.json` with `next_action: dispatch_to_subagent`, `current_iter: <prev+1>`.
- Determine the domain route from the dossier's data shape and the hypothesis's `concrete_change`. Routing rules (same as the existing router rules in `agents/ml-engineer.md`):
  - Image data (jpg/png/tif/bmp/webp/dcm/nii) OR vision model name → `cv-engineer`.
  - Text data with classify/tag/NER/extract task → `nlp-engineer`.
  - LLM/VLM finetune / DPO / GRPO / quantize / serve / merge → `llm-engineer`.
  - Tabular numeric/categorical CSV/parquet/xlsx → `ml-engineer`.
- Return the chosen hypothesis to the engine.

## Output format

Return to the engine:

```
SELECTED: <hypothesis id> — <one_line>
SOURCE: <literature | mutation | failure_mining | cross_domain | adversarial>
THEME: <theme>
EXPECTED_GAIN: <low | med | high>
EXPECTED_COST: <low | med | high>
SCORE: <number>
DOMAIN ROUTE: <ml-engineer | cv-engineer | nlp-engineer | llm-engineer>
NEXT: dispatch to <domain agent> with iteration_dir = <abs path>/iterations/<NNN>/
```

## Verification gates

Before returning to the engine, confirm:

- [ ] At least one candidate was eligible (live, within cost ceiling). If zero, the cost-ceiling output above is returned instead — that is the correct path, not a gate failure.
- [ ] The chosen hypothesis has `status: running` in the latest record of `hypotheses.jsonl`.
- [ ] `status.json` was updated with `next_action: dispatch_to_subagent` and `current_iter` incremented by 1.
- [ ] `iterations/<NNN>/` directory exists (create it as part of this step).
- [ ] The domain route matches the routing rules in this skill's Step 5.
- [ ] If a random tie-break occurred, the candidate's selection was logged with the reason in a note that `re-update-narrative` will fold into `narrative_delta.md` (write a `<workdir>/research_engine/iterations/<NNN>/selection_note.md` with the random-tie-break record).
- [ ] `<workdir>/research_engine/iterations/<NNN>/hypothesis.json` exists and is valid JSON containing the full hypothesis record.

If any gate fails, do not return to the engine — fix and re-verify.

## Hard constraints

- The score formula is the contract. Do not silently override it; if a different selection rule is needed, that's a spec change, not an in-skill judgment call.
- The diversity penalty (novelty_factor) is non-optional. Three same-theme runs in a row breaks the engine's exploration discipline.
- The cost ceiling check is non-optional. The engine's contract with the user is "you asked me to spend; I asked before exceeding the ceiling."
- Never select a hypothesis whose latest-version `status` is not `live`.
- Never overwrite `hypotheses.jsonl` records — always append new versioned records.
