---
name: re-mine-literature
description: Use to run a literature-mining pass during a research-engine session. Reads the narrative's most recent "Open questions" / "Currently suspected" entries and searches for evidence that would resolve or strengthen them. Writes claims into the narrative, not citation tables. Only fires from inside the research-engine agent. Do NOT invoke from ml-engineer, cv-engineer, nlp-engineer, llm-engineer, or from a user request.
license: MIT
metadata:
  source: ml-engineer
  version: 0.3.0
---

# Mine Literature

The engine's continuous reader. Pulls arxiv / Kaggle / HF cookbook / blogs / conference proceedings, biased by what the narrative most recently flagged as unknown. Wraps `dl-prior-art` and `ml-engineer-research` but reads the narrative *first* so the search is informed.

This is not a fixed search. It is a *narrative-driven* search: the queries change every pass because the narrative changes every iteration.

## When to invoke

- Once at engine start, immediately after `re-frame-problem`, to seed the initial narrative.
- Every K=3 iterations during the engine loop, biased by the latest narrative state.
- On demand when `re-generate-hypotheses` reports its candidate pool is exhausted.

## When NOT to invoke

- The same problem class was researched in this session less than 30 minutes ago AND the narrative has not changed in a way that affects search terms.
- A user one-shot question — engines don't run for one-shot questions.

## Process

### Step 1 — Read the narrative

Read `<workdir>/research_engine/narrative.md`. Extract:

- The 3 most recent entries in `## Open questions`.
- The 3 lowest-confidence entries in `## Currently suspected`.
- The 3 most recent entries in `## Ruled out` (these tell you what NOT to search for again).

If the narrative is empty (engine start), use the dossier's "Already known to work / fail" sections + the problem statement + metric to seed the initial search.

### Step 2 — Form queries

For each unknown identified in Step 1, form one query that targets winning solutions and recent (≤2 years) writeups. Reuse the query discipline from `dl-prior-art`:

- Include the data domain and task type.
- Include a competition or cookbook qualifier (`Kaggle 1st place`, `HuggingFace cookbook`, etc.) when applicable.
- Append `2024 OR 2025 OR 2026` for fast-moving topics.

Total query budget per pass: **at most 4 queries**. Cap so the pass is bounded.

### Step 3 — Search

Use `WebSearch` for each query. Keep the top 2-3 results per query.

### Step 4 — Read

Use `WebFetch` for the most promising results. Budget: **at most 6 fetches per pass.** Same prioritization as `dl-prior-art` (Kaggle 1st-place threads → HF cookbook → HF blog → grandmaster blogs → arxiv if directly relevant).

Skip:
- Tutorials with no real result.
- Marketing posts.
- Anything older than 3 years for fast-moving topics.

### Step 5 — Extract claims, not citations

For each fetched source, extract claims that map to one of three narrative sections:

- **Ruled out** — "X does not work for this data shape." (Strong claim; needs 2+ sources or one strongly-credible source.)
- **Currently suspected** — "X likely helps for this data shape." (Confidence: low / med / high based on source count and credibility.)
- **Open questions** — "Sources disagree on X" or "no sources cover X for this exact shape."

Each claim must be one sentence, attributable to a specific source URL, and independently testable.

### Step 6 — Write the reading file

Write `<workdir>/research_engine/reading/<TS>.md` where `<TS>` is the current UTC timestamp formatted as `YYYY-MM-DDTHH-MMZ` (ISO-8601 with `:` replaced by `-` and seconds dropped, matching the schema doc). Use the schema in `docs/superpowers/specs/research-engine-workdir-schema.md`. Required sections: header / `## Triggered by` / `## Queries used` / `## Sources read` / `## Claims extracted`.

### Step 7 — Append claims to narrative

Append (do not overwrite) to `<workdir>/research_engine/narrative.md`:

- Add new "Currently suspected" claims with confidence and `(reading <TS>: <one-line source summary>)`.
- Add new "Ruled out" claims with `(reading <TS>: <one-line source summary>)`.
- Add new "Open questions".

Do not duplicate. If a claim already exists in the narrative, increment its confidence by one tier (low → med, med → high) only if the new source is independent of the prior one. Otherwise leave it.

## Output format

Return to the engine:

```
LITERATURE PASS COMPLETE: <abs path to reading/<TS>.md>
NEW CLAIMS: <count>
  - Ruled out: <n>
  - Suspected: <n>
  - Open questions: <n>
QUERIES USED: <n>/4
FETCHES USED: <n>/6
NEXT: re-generate-hypotheses
```

## Verification gates

Before returning to the engine, confirm:

- [ ] `<workdir>/research_engine/reading/<TS>.md` exists, has the four required sections (`## Triggered by`, `## Queries used`, `## Sources read`, `## Claims extracted`), and lists at least 1 source under `## Sources read` (a pass with zero sources is suspicious — either re-run with broader queries or report `NEW CLAIMS: 0` honestly).
- [ ] `<workdir>/research_engine/narrative.md` was appended to (not overwritten); the original `## Ruled out`, `## Currently suspected`, `## Open questions` sections still contain their pre-pass entries.
- [ ] No "Currently suspected" entry was added with `confidence: high` from a single source.
- [ ] Query and fetch budgets were respected (≤4 queries, ≤6 fetches).
- [ ] No source older than 3 years was used for fast-moving topics (LLM, VLM, SAM family, instruction tuning).

If any gate fails, do not return to the engine — fix and re-verify.

## Hard constraints

- Maximum 4 queries and 6 fetches per pass. The cap is the contract — exceeding it breaks the engine's cost model.
- Never paste raw quotes into the narrative. Internalize and rewrite as one-sentence claims.
- Never add a "Currently suspected" with confidence: high from a single source. High requires 2+ independent sources.
- Skip sources older than 3 years for fast-moving topics (LLM, VLM, SAM family). Hard rule.
