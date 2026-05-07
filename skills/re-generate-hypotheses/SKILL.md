---
name: re-generate-hypotheses
description: Use to populate or refresh the live hypothesis list during a research-engine session. Generates candidates from five sources (literature, mutation of survivors, failure-mining, cross-domain analogy, adversarial wild-card) with non-zero quotas per source. Only fires from inside the research-engine agent. Do NOT invoke from ml-engineer, cv-engineer, nlp-engineer, llm-engineer, or from a user request.
license: MIT
metadata:
  source: ml-engineer
  version: 0.3.0
---

# Generate Hypotheses

Produce hypotheses that the engine will run. Each hypothesis is **testable in this engine** — it has a concrete change to make, runs through the existing transactional loop, and produces a metric comparable to the champion.

The defining discipline: **five sources, non-zero quota per source, every round.** The adversarial / cross-domain quotas are what keep the engine out of local optima.

## When to invoke

- Once at engine start, immediately after `re-mine-literature`'s seeding pass, to populate the initial live list (~20 candidates).
- Every iteration after `re-update-narrative`, to refresh the list given the new claim set (typically 3-5 new candidates).
- On demand from `re-detect-plateau` when it returns `continue` but flags low candidate diversity.

## When NOT to invoke

- The live list already has more than 30 unblocked candidates and the narrative has not changed since the last call.
- A user one-shot question — engines don't run for one-shot questions.

## Process

### Step 1 — Read inputs

- `<workdir>/research_engine/dossier.md` — for problem framing.
- `<workdir>/research_engine/narrative.md` — for current claims.
- `<workdir>/research_engine/hypotheses.jsonl` — for the live list (to avoid duplicates and to seed mutation).
- `<workdir>/research_engine/leaderboard.jsonl` — for failure mining.
- The most recent `<workdir>/research_engine/reading/<TS>.md` — for fresh literature claims.

### Step 2 — Generate from each source

The five sources, with their quota for the round (initial-seed quotas in parens):

#### (a) Literature — quota: 1+ per round (initial seed: 5+)

For each "Currently suspected" claim in the narrative or each "Claim extracted" in the latest reading file, propose one hypothesis that *tests* the claim. Concrete change must be a single, isolated modification to the current champion. Source field: `literature`.

#### (b) Mutation of survivors — quota: 1+ per round (initial seed: 0 — no survivors yet)

Read the top 5 hypotheses by leaderboard score. For each, propose one *perturbation*: change a hyperparameter, swap a component, scale magnitude. Lineage must be tracked: `parent_id` is set to the survivor's id. Source field: `mutation`.

#### (c) Failure mining — quota: 1+ per round (initial seed: 0 — no failures yet)

Read the bottom 5 hypotheses (failed or scored worse than baseline) on the leaderboard. For each, propose a *counter-hypothesis* — one that tests *why* the failure happened. Example: if "no augmentation → score dropped", a counter-hypothesis is "the wrong augmentation strength was used; try a milder version." Source field: `failure_mining`.

#### (d) Cross-domain analogy — quota: 1+ per round (initial seed: 3+)

Force an analogy from a *different* domain than the user's. Examples:
- User is doing medical imaging classification → analogy from satellite imagery, microscopy, or natural scenes.
- User is doing tabular regression → analogy from time-series forecasting or ranking.
- User is doing LLM finetuning → analogy from VLM, encoder-NLP, or speech recognition.

The analogy must be concrete: name the source domain, name the technique, propose how it transfers. Source field: `cross_domain`.

#### (e) Adversarial wild-card — quota: 1+ per round (initial seed: 2+, NEVER ZERO)

The "dumbest thing that might work" budget. Propose at least one hypothesis that contradicts the current narrative's "Currently suspected" entries OR tries the obviously-wrong thing on purpose. Examples:

- "The data is dirty — the metric is wrong; switch to a robust metric and re-baseline."
- "We've been using LoRA r=16; try r=2 just to see how much capacity we actually need."
- "Drop half the features randomly and re-train — if the score barely changes, we have a feature-quality problem."

This quota **is never zero**. Even at plateau. Even when literature is rich. Even when survivors are strong. It is the engine's anti-local-optimum reflex.

Source field: `adversarial`.

### Step 3 — De-duplicate

Reject any candidate whose `concrete_change` matches an existing live or archived hypothesis. Reject candidates whose only difference is wording.

Dedup rule: lowercase + strip whitespace + remove punctuation, then compare exact string. If two candidates collide under this rule, keep the older one and discard the new.

### Step 4 — Estimate gain and cost

For each surviving candidate, fill `expected_gain` ∈ `{low, med, high}` and `expected_cost` ∈ `{low, med, high}`:

- Gain = how much would the result, regardless of sign, change the narrative? (high if it ends a debate; low if it confirms the obvious.)
- Cost = expected wall-time + paid-remote dollars relative to the baseline run.

These are estimates. The selector uses them; they don't have to be perfect.

### Step 5 — Append to `hypotheses.jsonl`

Each new candidate is one JSONL line, with monotonically increasing `id` (`h_001`, `h_002`, ...) and `version: 1`. Use the schema in `docs/superpowers/specs/research-engine-workdir-schema.md`. The required fields per record: `id`, `version`, `created_iter`, `updated_iter`, `status` (always `live` for new candidates), `theme`, `one_line`, `concrete_change`, `source`, `parent_id` (null unless `mutation`), `expected_gain`, `expected_cost`, `rank` (initialize to a value larger than any current rank — `re-update-narrative` will re-rank later).

### Step 6 — Verify quotas were met

Reject the round if any source has zero candidates contributed. The engine should not proceed with a degenerate list. If a source is unable to produce a candidate (e.g., failure-mining at engine start with no failures yet), record a placeholder reason in the engine's status; do NOT silently zero-out the quota in steady state.

## Output format

Return to the engine:

```
HYPOTHESES GENERATED: <total count>
  - Literature: <n>
  - Mutation: <n>
  - Failure mining: <n>
  - Cross-domain: <n>
  - Adversarial: <n>
LIVE LIST SIZE: <total live count after this round>
NEXT: re-select-next
```

## Verification gates

Before returning to the engine, confirm:

- [ ] Every appended record in `hypotheses.jsonl` has all required fields (`id`, `version`, `created_iter`, `updated_iter`, `status`, `theme`, `one_line`, `concrete_change`, `source`, `parent_id`, `expected_gain`, `expected_cost`, `rank`).
- [ ] Every record's `source` is one of `{literature, mutation, failure_mining, cross_domain, adversarial}`.
- [ ] Every `mutation` record has a non-null `parent_id`.
- [ ] At least one record per source quota in steady state. At engine start (no survivors, no failures), quota exceptions for `mutation` and `failure_mining` are allowed and must be flagged in the output.
- [ ] **Adversarial count ≥ 1** in steady state, ≥ 2 at engine start. NEVER zero.
- [ ] **Cross-domain count ≥ 1** in any round. NEVER zero.
- [ ] No two new records have the same `concrete_change` after the dedup-rule normalization (lowercase + strip whitespace + strip punctuation).
- [ ] All `id` values are monotonically increasing relative to the prior `hypotheses.jsonl` max id.

If any gate fails, do not return to the engine — fix and re-verify.

## Hard constraints

- Adversarial quota is **NEVER ZERO** in steady state. Engine start may have an exempt round if everything is in literature/cross-domain.
- Cross-domain quota is **NEVER ZERO** in any round.
- Every hypothesis must have a `concrete_change` that fits in one sentence and is implementable through the existing skills.
- No hypothesis without a `source`. No hypothesis without a `theme`.
- De-duplication is exact-match on `concrete_change` after lowercase + strip whitespace + strip punctuation. Re-wording the same idea to dodge the dedup is a bug — fix the dedup, do not weaken it.
