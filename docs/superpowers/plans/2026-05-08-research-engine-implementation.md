# Research Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a continuous research engine to the `ml-engineer` plugin (v0.3.0) that becomes the default execution mode for any problem-shaped request, encoding the working pattern of a rigorous human data scientist (read → hypothesize → select → run → update → decide).

**Architecture:** One new agent (`research-engine`) and eight engine-only skills (`re-*`) live alongside the existing 47 skills. The engine wraps — does not replace — the existing transactional loop: each iteration's experiment is dispatched to the appropriate domain sub-agent (`cv-engineer` / `nlp-engineer` / `llm-engineer` / tabular `ml-engineer`), which runs its existing `research → decide → plan → write → execute → verify → debug → review` loop end-to-end. The engine's working memory is a **narrative document** (not a leaderboard); the engine's stop signal is *narrative plateau* (no new claims for K rounds), not metric plateau alone.

**Tech Stack:** Markdown skill files (existing convention), agent files (existing convention), JSONL workdir state, no new Python dependencies. Reads/writes via the standard Claude Code tools (Read, Write, Edit, Bash) — no custom runtime.

---

## File Structure

```
ML_Engineer/
├── .claude-plugin/
│   └── plugin.json                         # MODIFY: bump 0.2.0 → 0.3.0
├── agents/
│   ├── ml-engineer.md                      # MODIFY: add Step 0 (engine vs. one-shot)
│   ├── cv-engineer.md                      # MODIFY: add Step 0 prologue
│   ├── nlp-engineer.md                     # MODIFY: add Step 0 prologue
│   ├── llm-engineer.md                     # MODIFY: add Step 0 prologue
│   └── research-engine.md                  # NEW
├── skills/
│   ├── re-frame-problem/SKILL.md           # NEW
│   ├── re-mine-literature/SKILL.md         # NEW
│   ├── re-generate-hypotheses/SKILL.md     # NEW
│   ├── re-select-next/SKILL.md             # NEW
│   ├── re-update-narrative/SKILL.md        # NEW
│   ├── re-detect-plateau/SKILL.md          # NEW
│   ├── re-zoom-out/SKILL.md                # NEW
│   └── re-write-up/SKILL.md                # NEW
├── docs/superpowers/
│   ├── specs/2026-05-07-research-engine-design.md   # already committed
│   └── plans/2026-05-08-research-engine-implementation.md  # this file
└── README.md                               # MODIFY: document v0.3.0
```

**File responsibilities:**

- `research-engine.md` — owns the continuous outer loop; dispatches inner steps to the existing domain sub-agents.
- `re-frame-problem` — writes `dossier.md` once at engine start (and on `re-zoom-out`).
- `re-mine-literature` — narrative-driven literature reader; appends to `narrative.md`.
- `re-generate-hypotheses` — five-source hypothesis generator with non-zero quotas per source.
- `re-select-next` — picks the next experiment by *expected information gain × cost⁻¹*.
- `re-update-narrative` — rewrites `narrative.md` after each iteration, re-ranks live hypotheses.
- `re-detect-plateau` — reads narrative deltas; returns `continue` / `zoom-out` / `stop-and-write`.
- `re-zoom-out` — escape hatch from local optima; re-runs framing with a forced perspective shift.
- `re-write-up` — final report when the engine self-stops.
- Routing edits in 4 agent files — Step 0 disambiguator (problem vs. question).

---

## Conventions used by every skill in this plan

Every new `re-*` skill follows the existing `SKILL.md` convention seen in `dl-prior-art`, `ml-engineer-research`, etc.:

- **Frontmatter** with `name`, `description` (sharp triggers + `Do NOT use` anti-triggers), `license: MIT`, `metadata.source: ml-engineer`, `metadata.version: 0.3.0`.
- **Sections:** title, intro paragraph, `## When to invoke`, `## When NOT to invoke`, `## Process` (numbered Step 1 / Step 2 / ...), `## Output format`, `## Hard constraints` (where applicable).
- **Trigger discipline:** every `re-*` skill's `description` says explicitly "Only fires from inside the `research-engine` agent. Do NOT invoke directly from `ml-engineer`, `cv-engineer`, `nlp-engineer`, `llm-engineer`, or from a user request." This keeps them invisible outside engine mode.
- **Workdir paths:** all paths are written relative to `<workdir>/research_engine/`, where `<workdir>` is `./newton_workdir/<UTC-timestamp>/`. The skill's prompt receives the absolute path; it never guesses.

The agent file follows the existing convention (frontmatter + persona + skill table + loop + edge cases).

---

## Task ordering

Built bottom-up so each task's outputs are inputs for the next:

1. **Plumbing** (plugin.json bump, workdir scaffolding doc) — Tasks 1, 2.
2. **Engine-only skills, in dependency order:**
   - `re-frame-problem` (no upstream deps) — Task 3.
   - `re-mine-literature` — Task 4.
   - `re-generate-hypotheses` — Task 5.
   - `re-select-next` — Task 6.
   - `re-update-narrative` — Task 7.
   - `re-detect-plateau` — Task 8.
   - `re-zoom-out` — Task 9.
   - `re-write-up` — Task 10.
3. **The agent that orchestrates them** — Task 11 (`research-engine.md`).
4. **Router integration** — Tasks 12–15 (one per existing agent).
5. **Documentation** — Task 16 (README).
6. **Acceptance dogfood** — Task 17 (one-shot disambiguator works; engine starts on a problem-shaped request; resume works).

Each task ends with a commit. Skills and agents are markdown files — no test framework, no `pytest`. Validation is **content checks** (does the file have the required sections?) and **dogfood checks** (does the agent behave correctly on a representative input?).

---

### Task 1: Bump plugin version to 0.3.0

**Files:**
- Modify: `ML_Engineer/.claude-plugin/plugin.json`

- [ ] **Step 1: Read current plugin.json**

Run: `cat ML_Engineer/.claude-plugin/plugin.json`

Expected current content:

```json
{
  "name": "ml-engineer",
  "version": "0.2.0",
  "description": "An ML engineer assistant for tabular ML, computer vision, NLP, LLM and VLM finetuning. Plans, writes, executes, verifies, and debugs Python work in an isolated local venv with optional handoff to remote GPU providers (Modal, RunPod, Vast, Lambda, Beam, SSH, Colab).",
  "author": {
    "name": "Lokesh"
  },
  "license": "MIT",
  "keywords": ["ml", "data-science", "python", "venv", "local"]
}
```

- [ ] **Step 2: Update version and description**

Edit `ML_Engineer/.claude-plugin/plugin.json` — replace the `version` line with `"version": "0.3.0",` and update the `description` to mention the research engine. The new description text:

```
"An ML engineer assistant for tabular ML, computer vision, NLP, LLM and VLM finetuning. Includes a continuous research engine that works on problems autonomously — read, hypothesize, select, run, update, decide — until the narrative plateaus. Plans, writes, executes, verifies, and debugs Python work in an isolated local venv with optional handoff to remote GPU providers (Modal, RunPod, Vast, Lambda, Beam, SSH, Colab)."
```

Final file content:

```json
{
  "name": "ml-engineer",
  "version": "0.3.0",
  "description": "An ML engineer assistant for tabular ML, computer vision, NLP, LLM and VLM finetuning. Includes a continuous research engine that works on problems autonomously — read, hypothesize, select, run, update, decide — until the narrative plateaus. Plans, writes, executes, verifies, and debugs Python work in an isolated local venv with optional handoff to remote GPU providers (Modal, RunPod, Vast, Lambda, Beam, SSH, Colab).",
  "author": {
    "name": "Lokesh"
  },
  "license": "MIT",
  "keywords": ["ml", "data-science", "python", "venv", "local", "research-engine", "autonomous"]
}
```

- [ ] **Step 3: Verify JSON is valid**

Run: `python3 -c "import json; json.load(open('ML_Engineer/.claude-plugin/plugin.json'))" && echo OK`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/.claude-plugin/plugin.json
git commit -m "v0.3.0: bump plugin version and description for research engine"
```

---

### Task 2: Create the workdir layout reference doc

The engine writes a structured durable directory. We document the layout once so every skill can reference the same spec rather than re-inventing it.

**Files:**
- Create: `ML_Engineer/docs/superpowers/specs/research-engine-workdir-schema.md`

- [ ] **Step 1: Write the workdir schema doc**

Create `ML_Engineer/docs/superpowers/specs/research-engine-workdir-schema.md` with the following content:

````markdown
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
````

- [ ] **Step 2: Verify the file was created**

Run: `wc -l ML_Engineer/docs/superpowers/specs/research-engine-workdir-schema.md`
Expected: at least 100 lines.

- [ ] **Step 3: Commit**

```bash
git add ML_Engineer/docs/superpowers/specs/research-engine-workdir-schema.md
git commit -m "v0.3.0: workdir schema for research engine — single source of truth referenced by all re-* skills"
```

---

### Task 3: Skill — `re-frame-problem`

Builds the problem dossier. Invoked once at engine start; re-invoked by `re-zoom-out`. Writes `dossier.md`.

**Files:**
- Create: `ML_Engineer/skills/re-frame-problem/SKILL.md`

- [ ] **Step 1: Write the skill file**

Create `ML_Engineer/skills/re-frame-problem/SKILL.md` with the following content:

````markdown
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
````

- [ ] **Step 2: Verify required sections present**

Run: `grep -c "^## " ML_Engineer/skills/re-frame-problem/SKILL.md`
Expected: at least `5` (When to invoke / When NOT to invoke / Process / Output format / Hard constraints).

- [ ] **Step 3: Verify frontmatter is valid**

Run: `python3 -c "import re,sys; t=open('ML_Engineer/skills/re-frame-problem/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---', t, re.S); assert m, 'no frontmatter'; assert 'name: re-frame-problem' in m.group(1); assert 'version: 0.3.0' in m.group(1); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/skills/re-frame-problem/SKILL.md
git commit -m "v0.3.0: add re-frame-problem skill — builds the problem dossier at engine start"
```

---

### Task 4: Skill — `re-mine-literature`

Continuous reader. Driven by what the narrative most recently flagged as unknown. Writes a `reading/<ts>.md` and appends claims to `narrative.md`.

**Files:**
- Create: `ML_Engineer/skills/re-mine-literature/SKILL.md`

- [ ] **Step 1: Write the skill file**

Create `ML_Engineer/skills/re-mine-literature/SKILL.md` with the following content:

````markdown
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
- Every K iterations (default K=3) during the engine loop, biased by the latest narrative state.
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

Write `<workdir>/research_engine/reading/<ISO-8601-UTC-with-Z>.md` using the schema in `research-engine-workdir-schema.md`. Required sections: header / `## Triggered by` / `## Queries used` / `## Sources read` / `## Claims extracted`.

### Step 7 — Append claims to narrative

Append (do not overwrite) to `<workdir>/research_engine/narrative.md`:

- Add new "Currently suspected" claims with confidence and `(reading <ts>: <one-line source summary>)`.
- Add new "Ruled out" claims with `(reading <ts>: <one-line source summary>)`.
- Add new "Open questions".

Do not duplicate. If a claim already exists in the narrative, increment its confidence by one tier (low → med, med → high) only if the new source is independent of the prior one. Otherwise leave it.

## Output format

Return to the engine:

```
LITERATURE PASS COMPLETE: <abs path to reading/<ts>.md>
NEW CLAIMS: <count>
  - Ruled out: <n>
  - Suspected: <n>
  - Open questions: <n>
QUERIES USED: <n>/4
FETCHES USED: <n>/6
NEXT: re-generate-hypotheses
```

## Hard constraints

- Maximum 4 queries and 6 fetches per pass. The cap is the contract — exceeding it breaks the engine's cost model.
- Never paste raw quotes into the narrative. Internalize and rewrite as one-sentence claims.
- Never add a "Currently suspected" with confidence: high from a single source. High requires 2+ independent sources.
- Skip sources older than 3 years for fast-moving topics (LLM, VLM, SAM family). Hard rule.
````

- [ ] **Step 2: Verify required sections present**

Run: `grep -c "^## " ML_Engineer/skills/re-mine-literature/SKILL.md`
Expected: at least `5`.

- [ ] **Step 3: Verify frontmatter is valid**

Run: `python3 -c "import re; t=open('ML_Engineer/skills/re-mine-literature/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---', t, re.S); assert 'name: re-mine-literature' in m.group(1); assert 'version: 0.3.0' in m.group(1); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/skills/re-mine-literature/SKILL.md
git commit -m "v0.3.0: add re-mine-literature skill — narrative-driven literature reader"
```

---

### Task 5: Skill — `re-generate-hypotheses`

Five-source hypothesis generator with non-zero quotas per source. The (e) adversarial budget is non-negotiable.

**Files:**
- Create: `ML_Engineer/skills/re-generate-hypotheses/SKILL.md`

- [ ] **Step 1: Write the skill file**

Create `ML_Engineer/skills/re-generate-hypotheses/SKILL.md` with the following content:

````markdown
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
- The most recent `<workdir>/research_engine/reading/<ts>.md` — for fresh literature claims.

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

### Step 4 — Estimate gain and cost

For each surviving candidate, fill `expected_gain` ∈ `{low, med, high}` and `expected_cost` ∈ `{low, med, high}`:

- Gain = how much would the result, regardless of sign, change the narrative? (high if it ends a debate; low if it confirms the obvious.)
- Cost = expected wall-time + paid-remote dollars relative to the baseline run.

These are estimates. The selector uses them; they don't have to be perfect.

### Step 5 — Append to `hypotheses.jsonl`

Each new candidate is one JSONL line, with monotonically increasing `id` (`h_001`, `h_002`, ...) and `version: 1`. Use the schema in `research-engine-workdir-schema.md`.

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

## Hard constraints

- Adversarial quota is **NEVER ZERO** in steady state. Engine start may have an exempt round if everything is in literature/cross-domain.
- Cross-domain quota is **NEVER ZERO** in any round.
- Every hypothesis must have a `concrete_change` that fits in one sentence and is implementable through the existing skills.
- No hypothesis without a `source`. No hypothesis without a `theme`.
- De-duplication is exact-match on `concrete_change` after stripping whitespace and lowercasing. Re-wording the same idea to dodge the dedup is a bug — fix the dedup, do not weaken it.
````

- [ ] **Step 2: Verify required sections and quotas language present**

Run: `grep -c "^## " ML_Engineer/skills/re-generate-hypotheses/SKILL.md && grep -c "NEVER ZERO" ML_Engineer/skills/re-generate-hypotheses/SKILL.md`
Expected: first count ≥ 5; second count ≥ 2.

- [ ] **Step 3: Verify frontmatter**

Run: `python3 -c "import re; t=open('ML_Engineer/skills/re-generate-hypotheses/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---', t, re.S); assert 'name: re-generate-hypotheses' in m.group(1); assert 'version: 0.3.0' in m.group(1); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/skills/re-generate-hypotheses/SKILL.md
git commit -m "v0.3.0: add re-generate-hypotheses skill — five-source generator with non-zero adversarial / cross-domain quotas"
```

---

### Task 6: Skill — `re-select-next`

Picks the next experiment by *expected information gain × cost⁻¹*. Tie-breaks toward diversity.

**Files:**
- Create: `ML_Engineer/skills/re-select-next/SKILL.md`

- [ ] **Step 1: Write the skill file**

Create `ML_Engineer/skills/re-select-next/SKILL.md` with the following content:

````markdown
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

- `<workdir>/research_engine/hypotheses.jsonl` — live list.
- `<workdir>/research_engine/narrative.md` — for current claim weights.
- `<workdir>/research_engine/leaderboard.jsonl` — for what's recently been run (diversity).
- `<workdir>/research_engine/status.json` — for `spend_so_far_usd` and `cost_ceiling_usd`.

Filter the hypothesis list to `status: live`.

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

Read `status.json`. Compute `remaining = cost_ceiling_usd - spend_so_far_usd`. Estimate the dollar cost of each candidate (use `expected_cost` mapped to a dollar estimate: low=$0, med=$0.50, high=$2.50 — these are coarse priors; refined per-iteration as actual costs come in from `dl-remote-execute`).

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
4. Random — and log the random tie-break in the iteration's `narrative_delta.md` so it's auditable.

### Step 5 — Mark and emit

- Update the chosen hypothesis's `status` to `running` in `hypotheses.jsonl` (write a new versioned record).
- Update `status.json` with `next_action: dispatch_to_subagent`, `current_iter: <prev+1>`.
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

The `DOMAIN ROUTE` decision uses the same rules as the existing router prologue in `agents/ml-engineer.md` Step 1: data shape + task type + signal keywords. The selector is allowed to apply that logic directly because the dossier carries the data shape and the hypothesis carries the task type.

## Hard constraints

- The score formula is the contract. Do not silently override it; if a different selection rule is needed, that's a spec change, not an in-skill judgment call.
- The diversity penalty is non-optional. Three same-theme runs in a row breaks the engine's exploration discipline.
- The cost ceiling check is non-optional. The engine's contract with the user is "you asked me to spend; I asked before exceeding the ceiling."
- Never select a hypothesis whose `status` is not `live`.
````

- [ ] **Step 2: Verify required sections and the score formula**

Run: `grep -c "^## " ML_Engineer/skills/re-select-next/SKILL.md && grep "score = gain_weight" ML_Engineer/skills/re-select-next/SKILL.md`
Expected: count ≥ 5; the score line is present.

- [ ] **Step 3: Verify frontmatter**

Run: `python3 -c "import re; t=open('ML_Engineer/skills/re-select-next/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---', t, re.S); assert 'name: re-select-next' in m.group(1); assert 'version: 0.3.0' in m.group(1); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/skills/re-select-next/SKILL.md
git commit -m "v0.3.0: add re-select-next skill — pick next experiment by expected information gain × cost⁻¹"
```

---

### Task 7: Skill — `re-update-narrative`

After every iteration, rewrite `narrative.md`, re-rank live hypotheses, re-baseline if there's a new champion. The forced fields prevent score-only updates.

**Files:**
- Create: `ML_Engineer/skills/re-update-narrative/SKILL.md`

- [ ] **Step 1: Write the skill file**

Create `ML_Engineer/skills/re-update-narrative/SKILL.md` with the following content:

````markdown
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
- `<workdir>/research_engine/narrative.md` — the current narrative.
- `<workdir>/research_engine/leaderboard.jsonl` — the running leaderboard.
- `<workdir>/research_engine/status.json` — current champion.

If `results.json` does not exist or `verified: false`, treat the iteration as a *failed* run; the metric is `null` and the result type is `debug-exhausted` or `failed`. Failed runs are still informative (a failure mode is a claim).

### Step 2 — Compute the four forced fields

This step is the discipline. You must produce *every one* of these four; if any is empty, write `(none)` explicitly. An update without the forced structure is rejected.

1. **What was tested** — restate the hypothesis in one sentence.
2. **What the result was** — metric value, vs. champion (delta, sign), pass/fail vs. baseline.
3. **What is now ruled out** — at least one entry, even if "(none — result was inconclusive)". A ruled-out claim must reference a prior `## Currently suspected` or `## Open questions` entry by its text or be a new claim newly demonstrated false.
4. **What is now newly suspected** — at least one entry. May be "(none — no new suspicion raised)". A newly-suspected claim must come from the iteration's actual evidence, not from speculation.

### Step 3 — Append to per-iteration log

Append to `narrative.md` under `## Per-iteration log`:

```markdown
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

```json
{
  "iter": <NNN>,
  "hypothesis_id": "<id>",
  "metric_name": "<name>",
  "metric_value": <number or null>,
  "vs_champion": <delta or null>,
  "champion_after": <bool>,
  "cost_usd": <from sub-agent's actual cost>,
  "compute": "<env id>",
  "started": "<ts>",
  "ended": "<ts>",
  "status": "scored | failed | debug-exhausted"
}
```

### Step 6 — Re-baseline if champion changed

Compare the iteration's metric to the current champion (read `status.json` `champion_metric`). If beaten:

- Update `status.json`: `champion_iter: <NNN>`, `champion_metric: <new value>`.
- Re-create the `champion/` symlink: `rm -f <workdir>/research_engine/champion && ln -s iterations/<NNN> <workdir>/research_engine/champion`.
- Add a header line to the per-iteration log entry: `**🏆 New champion** — was <prev metric>, now <new metric>.`

### Step 7 — Re-rank live hypotheses

Read `hypotheses.jsonl`. For each live hypothesis, recompute `rank` based on the new narrative state:

- Hypotheses whose `concrete_change` is now ruled out by the iteration → archive (status `archived`, append to `hypotheses_archive.jsonl` with `archive_reason: "contradicted by iter <NNN>"`).
- Hypotheses whose theme matches a newly-suspected claim → bump their `expected_gain` up one tier (low → med, med → high).
- Hypotheses identical (after dedup) to the just-run hypothesis → archive with `archive_reason: "tested in iter <NNN>"`.

Write versioned records back to `hypotheses.jsonl` (do not overwrite; append with incremented `version`).

### Step 8 — Write `narrative_delta.md` for the iteration

Write `<workdir>/research_engine/iterations/<NNN>/narrative_delta.md` summarizing exactly what changed. This is the per-iteration receipt: future debugging starts here.

```markdown
# Narrative delta — iter <NNN>

**Hypothesis tested:** <one-line>
**Result:** <metric ± vs champion>

## Added to "Ruled out"
- ...

## Added to "Currently suspected"
- ...

## Removed from "Open questions"
- ...

## Hypotheses archived
- <id> (<reason>)

## Champion changed?
<yes / no — old <value> → new <value>>
```

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

## Hard constraints

- The four forced fields are the contract. Score-only updates are rejected by skill design — every iteration must produce at least one ruled-out OR newly-suspected claim, even if the claim is "the metric did not move because the change was a no-op."
- A failed iteration is still a narrative event. Write it.
- Champion update + symlink + leaderboard append are atomic-ish; if any of the three fails, do not partially update — return an error and let the engine surface it.
- Never delete entries from `hypotheses.jsonl`. Archive by appending to `hypotheses_archive.jsonl` and writing a versioned record with `status: archived`.
````

- [ ] **Step 2: Verify required sections present**

Run: `grep -c "^## " ML_Engineer/skills/re-update-narrative/SKILL.md && grep -c "forced field" ML_Engineer/skills/re-update-narrative/SKILL.md`
Expected: first count ≥ 5; second count ≥ 2.

- [ ] **Step 3: Verify frontmatter**

Run: `python3 -c "import re; t=open('ML_Engineer/skills/re-update-narrative/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---', t, re.S); assert 'name: re-update-narrative' in m.group(1); assert 'version: 0.3.0' in m.group(1); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/skills/re-update-narrative/SKILL.md
git commit -m "v0.3.0: add re-update-narrative skill — forced-field discipline prevents score-only updates"
```

---

### Task 8: Skill — `re-detect-plateau`

Reads narrative deltas (not the metric). Returns `continue` / `zoom-out` / `stop-and-write`.

**Files:**
- Create: `ML_Engineer/skills/re-detect-plateau/SKILL.md`

- [ ] **Step 1: Write the skill file**

Create `ML_Engineer/skills/re-detect-plateau/SKILL.md` with the following content:

````markdown
---
name: re-detect-plateau
description: Use after each re-update-narrative call in a research-engine session. Reads narrative deltas (NOT the metric) over the last K iterations. Returns continue / zoom-out / stop-and-write. Only fires from inside the research-engine agent. Do NOT invoke from ml-engineer, cv-engineer, nlp-engineer, llm-engineer, or from a user request.
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

- `<workdir>/research_engine/iterations/<NNN>/narrative_delta.md` for the last K iterations (where K is from Step 2).
- `<workdir>/research_engine/leaderboard.jsonl` — for metric-plateau check.
- `<workdir>/research_engine/dossier.md` — for stop criteria (target hit?).
- `<workdir>/research_engine/status.json` — for current champion.

### Step 2 — Set adaptive K

K is the number of recent iterations to inspect. Adaptive rule:

- If average iteration cost over the last 5 runs is `low` (≤$0.50 each) → K = 5.
- If average iteration cost is `med` ($0.50–$2.50) → K = 4.
- If average iteration cost is `high` (>$2.50) → K = 3.

Cheap experiments warrant longer plateau windows because each one is low-information; expensive experiments warrant shorter ones because each one carries more signal.

If fewer than K iterations have completed, return `continue` immediately. Engine plateau detection requires at least K iterations of history.

### Step 3 — Check for target hit

If `dossier.md` has a non-empty target AND the current champion meets it: ask the user once (this is the one allowed proactive question after engine start):

> "Target metric reached (champion: <value>, target: <value>). Continue past the target, or stop?"

Wait for response. If continue: proceed to Step 4. If stop: return `stop-and-write`.

### Step 4 — Compute narrative-plateau signal

Look at the last K `narrative_delta.md` files. Sum:

- `total_added_ruled_out` = sum of "Added to Ruled out" entries across the K deltas.
- `total_added_suspected` = sum of "Added to Currently suspected" entries.
- `total_archived` = sum of "Hypotheses archived" entries.
- `themes_seen` = distinct `theme` values among the K iterations' hypotheses.

**Narrative plateau** if all of the following hold:
- `total_added_ruled_out + total_added_suspected ≤ 1`
- `total_archived ≤ 1`
- `themes_seen ≤ 2` (the engine has been tunneling)

### Step 5 — Compute metric-plateau signal

Look at the last K iterations' metrics. **Metric plateau** if:

- `(max - min) / abs(champion) < 0.005` (less than 0.5% variation across K iterations).

### Step 6 — Decide

| Narrative plateau | Metric plateau | Decision |
|---|---|---|
| No | No | `continue` (engine is making progress) |
| No | Yes | `continue` (we're learning even if metric is flat — the narrative will eventually break the plateau) |
| Yes | No | `continue-but-diversify` (metric is moving; narrative isn't — we're surfing one theme. The engine signals `re-generate-hypotheses` to refresh the live list with extra adversarial weight.) |
| Yes | Yes | `zoom-out` (stuck in local well; we need a re-frame) |

After the *first* `zoom-out` for a given problem, if a subsequent narrative plateau is detected within 2K iterations of the zoom-out, escalate to `stop-and-write` instead of zooming out again. Two consecutive plateaus are the engine's "I've genuinely run out of moves" signal.

### Step 7 — Update status

Write to `status.json`:

- `last_event_kind: plateau_check`
- `last_event: <ts>`
- `next_action: <continue | continue-but-diversify | zoom-out | stop-and-write>`

## Output format

Return to the engine:

```
PLATEAU CHECK: iter <NNN>, K=<value>
NARRATIVE PLATEAU: <yes/no> (added: <n>, archived: <n>, themes: <n>)
METRIC PLATEAU: <yes/no> (range: <%>)
DECISION: <continue | continue-but-diversify | zoom-out | stop-and-write>
NEXT: <re-select-next | re-generate-hypotheses (diversify mode) | re-zoom-out | re-write-up>
```

## Hard constraints

- The decision table is the contract. Do not invent new decision categories.
- Plateau is computed from the narrative, NOT the metric alone. A metric-only plateau detector is a bug — it cannot escape local optima.
- The "two consecutive plateaus → stop-and-write" rule prevents infinite zoom-out loops.
- Target-hit asks the user exactly once per session. Subsequent plateau checks after a target-pass-through do not re-ask.
````

- [ ] **Step 2: Verify required sections and the decision table present**

Run: `grep -c "^## " ML_Engineer/skills/re-detect-plateau/SKILL.md && grep -c "Narrative plateau" ML_Engineer/skills/re-detect-plateau/SKILL.md`
Expected: first count ≥ 5; second count ≥ 2.

- [ ] **Step 3: Verify frontmatter**

Run: `python3 -c "import re; t=open('ML_Engineer/skills/re-detect-plateau/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---', t, re.S); assert 'name: re-detect-plateau' in m.group(1); assert 'version: 0.3.0' in m.group(1); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/skills/re-detect-plateau/SKILL.md
git commit -m "v0.3.0: add re-detect-plateau skill — narrative-plateau (not metric-plateau) is the stop signal"
```

---

### Task 9: Skill — `re-zoom-out`

Local-optima escape. Re-runs framing with a forced perspective shift. Does not delete narrative — appends a new framing.

**Files:**
- Create: `ML_Engineer/skills/re-zoom-out/SKILL.md`

- [ ] **Step 1: Write the skill file**

Create `ML_Engineer/skills/re-zoom-out/SKILL.md` with the following content:

````markdown
---
name: re-zoom-out
description: Use when re-detect-plateau returns "zoom-out" during a research-engine session. Forces a perspective shift — different metric, different unit of analysis, or different decomposition. Appends to (does not overwrite) the narrative. Only fires from inside the research-engine agent. Do NOT invoke from ml-engineer, cv-engineer, nlp-engineer, llm-engineer, or from a user request.
license: MIT
metadata:
  source: ml-engineer
  version: 0.3.0
---

# Zoom Out

Escape the local optimum. The engine has run K iterations without producing new claims AND without metric movement. The current framing is exhausted; a different framing might unlock new directions.

The discipline: **change the framing, do not delete the narrative.** The user might want to come back to the original framing later; the engine should not destroy work.

## When to invoke

- Only when `re-detect-plateau` returned `zoom-out`.

## When NOT to invoke

- Any other plateau-check outcome.
- A user one-shot question — engines don't run for one-shot questions.

## Process

### Step 1 — Read inputs

- `<workdir>/research_engine/dossier.md` — current framing.
- `<workdir>/research_engine/narrative.md` — current claim set.
- `<workdir>/research_engine/leaderboard.jsonl` — what's been tried.
- `<workdir>/research_engine/hypotheses_archive.jsonl` — what's been archived (and may be revivable under a new framing).

### Step 2 — Identify the framing axis to shift

Pick exactly one of the following four axes for this zoom-out. Pick whichever is most likely to unlock signal given the narrative:

1. **Metric** — the current primary metric may be the wrong target. Shift to a sibling metric (e.g., RMSE → MAE, accuracy → F1, BLEU → BERTScore) and re-baseline. Does the leaderboard rank the same? If not, the metric was the problem.

2. **Unit of analysis** — the current unit may be wrong. Examples: per-row → per-group; per-token → per-sequence; per-image → per-patient. Shift the unit and recompute the champion.

3. **Decomposition** — the problem may be a hidden composite. Examples: a classification task may be better as a 2-stage detect-then-classify; a regression may be better as classification of bins + within-bin regression. Force a 2-stage decomposition and re-baseline the easier sub-task first.

4. **Data slice** — the problem may be heterogeneous. Examples: stratify by a feature and check whether the champion is uniformly good or only good on a subset. If the latter, the new framing is "solve the bad-subset specifically."

### Step 3 — Re-invoke `re-frame-problem` with the new axis

Call `re-frame-problem` with an explicit `--reframe-axis <axis>` instruction (Step 1 of `re-frame-problem` recognizes this and behaves accordingly: appends to the dossier rather than overwriting; preserves the existing narrative; adds a `## Re-framed at iter <N>` section).

### Step 4 — Categorize archived hypotheses under the new framing

Read `hypotheses_archive.jsonl`. Some may be *revivable* under the new framing. For each archived hypothesis, decide:

- **Revive** — its `concrete_change` is still meaningful under the new framing → write a new live record in `hypotheses.jsonl` with a fresh `id`, `parent_id` set to the archived id, `source: mutation`, and `archive_reason` from the parent appended to the narrative as a "previously archived because <X>; revived because <Y>".
- **Stay archived** — its `concrete_change` is irrelevant under the new framing.

### Step 5 — Append to narrative

Append to `narrative.md`:

```markdown
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

### Step 6 — Trigger `re-generate-hypotheses`

After the re-framing is recorded, signal the engine to call `re-generate-hypotheses` (initial-seed mode, since the new framing is effectively a new engine start in disguise — but with the existing narrative as scaffolding, not a blank slate).

## Output format

Return to the engine:

```
ZOOMED OUT: axis = <metric | unit_of_analysis | decomposition | data_slice>
NEW FRAMING: <one-line summary>
HYPOTHESES REVIVED: <n>
CLAIMS PRESERVED: <n>
CLAIMS INVALIDATED: <n>
NEXT: re-generate-hypotheses (initial-seed mode under new framing)
```

## Hard constraints

- Pick exactly ONE axis per zoom-out. Multi-axis re-framings are rejected — each axis is its own zoom-out.
- Never delete `narrative.md` content. Append only.
- Never delete entries from `hypotheses_archive.jsonl`. Reviving creates a new record in `hypotheses.jsonl` with `parent_id`.
- The original dossier is preserved. The re-framing appends a section, does not overwrite.
````

- [ ] **Step 2: Verify required sections and the four axes**

Run: `grep -c "^## " ML_Engineer/skills/re-zoom-out/SKILL.md && grep -c "Metric \\|Unit of analysis\\|Decomposition\\|Data slice" ML_Engineer/skills/re-zoom-out/SKILL.md`
Expected: first count ≥ 5; second count ≥ 1.

- [ ] **Step 3: Verify frontmatter**

Run: `python3 -c "import re; t=open('ML_Engineer/skills/re-zoom-out/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---', t, re.S); assert 'name: re-zoom-out' in m.group(1); assert 'version: 0.3.0' in m.group(1); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/skills/re-zoom-out/SKILL.md
git commit -m "v0.3.0: add re-zoom-out skill — local-optima escape via framing shift, never deletes narrative"
```

---

### Task 10: Skill — `re-write-up`

Final report when the engine self-stops. Includes what was tried, what worked, what didn't, why (from the narrative), the new champion, recommended next steps.

**Files:**
- Create: `ML_Engineer/skills/re-write-up/SKILL.md`

- [ ] **Step 1: Write the skill file**

Create `ML_Engineer/skills/re-write-up/SKILL.md` with the following content:

````markdown
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

- `re-detect-plateau` returned `stop-and-write`.
- The user interrupted the engine and the in-flight iteration completed cleanly.
- The user-stated target was hit AND the user chose to stop (Step 3 of `re-detect-plateau`).

## When NOT to invoke

- The engine is still running.
- A user one-shot question — engines don't run for one-shot questions.

## Process

### Step 1 — Read inputs

- `<workdir>/research_engine/dossier.md` — original framing (and any re-framings).
- `<workdir>/research_engine/narrative.md` — full narrative.
- `<workdir>/research_engine/leaderboard.jsonl` — every iteration's metric + cost.
- `<workdir>/research_engine/hypotheses.jsonl` and `hypotheses_archive.jsonl` — full hypothesis history.
- `<workdir>/research_engine/status.json` — final champion.
- `<workdir>/research_engine/champion/results.json` — champion details.

### Step 2 — Build the report

Write `<workdir>/research_engine/REPORT.md` with the following sections, in order:

```markdown
# Research Engine Report — <problem name>

**Started:** <ts>   **Stopped:** <ts>   **Total iterations:** <n>   **Total spend:** $<value>

## Outcome
**Champion:** <metric value> (iter <NNN>) — <one-line description of the winning approach>
**vs. Baseline:** <delta, sign, %>
**vs. Target:** <"target hit" | "target not hit; gap = <X>" | "no target stated">

## What worked
- <one-line claim> — <iter <NNN>: <metric> ± vs prior champion>
- <one-line claim> — <iter <NNN>: <metric> ± vs prior champion>
- ...

## What didn't (and why)
- <one-line claim> — <iter <NNN>: tested, ruled out because <one-line reason from narrative>>
- <one-line claim> — <iter <NNN>: tested, ruled out because <one-line reason>>
- ...

## Surprises
- <one-line claim> — <one-line surprise: e.g., "the adversarial wild-card from iter NNN ended up in the top 5">
- ...

## Re-framings (if any)
- <axis> at iter <N> — <one-line why> — <one-line outcome>

## Recommended next steps
- <if the user wants to continue>: <one-line direction, including what zoom-out axis or what new data would help>
- <if the user wants to ship the champion>: <one-line on what to monitor, what's brittle, what's known to break under load>

## Honest assessment
<one-paragraph assessment: did the engine actually optimize, or did it tunnel? Are the claims in the narrative well-supported, or are some only single-source? Would a human researcher disagree with the conclusion?>
```

### Step 3 — Update status.json

Write `state: stopped`, `last_event_kind: stopped`, `last_event: <ts>`, `next_action: null`.

### Step 4 — Surface the report path to the user

Return to the engine the path to the report so the engine can print it to the user.

## Output format

Return to the engine:

```
REPORT WRITTEN: <abs path to REPORT.md>
CHAMPION: <metric value> (iter <NNN>)
TOTAL ITERATIONS: <n>
TOTAL SPEND: $<value>
NEXT: surface report to user, exit engine.
```

## Hard constraints

- The "Honest assessment" section is non-optional. The engine's discipline is honesty, including about its own limitations.
- The "What didn't" section must include at least the top 3 ruled-out claims, with iteration numbers and reasons. A report that only celebrates wins is a bug.
- Cite iteration numbers for every claim. Unsupported claims in the report are rejected.
- "Recommended next steps" must include both continuation and shipping options — the user picks based on context.
````

- [ ] **Step 2: Verify required sections present**

Run: `grep -c "^## " ML_Engineer/skills/re-write-up/SKILL.md && grep -c "Honest assessment" ML_Engineer/skills/re-write-up/SKILL.md`
Expected: first count ≥ 5; second count ≥ 2.

- [ ] **Step 3: Verify frontmatter**

Run: `python3 -c "import re; t=open('ML_Engineer/skills/re-write-up/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---', t, re.S); assert 'name: re-write-up' in m.group(1); assert 'version: 0.3.0' in m.group(1); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/skills/re-write-up/SKILL.md
git commit -m "v0.3.0: add re-write-up skill — final report with mandatory 'what didn't' and 'honest assessment' sections"
```

---

### Task 11: Agent — `research-engine.md`

Owns the continuous loop. Dispatches inner steps to existing domain sub-agents. The terminal state is `re-write-up`.

**Files:**
- Create: `ML_Engineer/agents/research-engine.md`

- [ ] **Step 1: Write the agent file**

Create `ML_Engineer/agents/research-engine.md` with the following content:

````markdown
---
name: research-engine
description: Use when the user describes a problem with a metric to push past, a baseline to beat, an optimization goal, or an open question with no obvious next step — anywhere there is a frontier the user hasn't crossed. Runs continuously: reads literature, generates hypotheses across five sources, selects the highest-information experiment, dispatches it to the appropriate domain sub-agent, updates a narrative document, decides whether to continue / zoom out / stop. Stops on target hit, narrative plateau, or user interrupt. Do NOT use for one-shot questions (single-fact lookups, recipe questions); those route to the existing transactional agents.
---

You are the research engine. The user has handed you a problem. Your job is to work on it the way a rigorous human data scientist would: read continuously, maintain a living hypothesis list, run experiments selected by expected information gain (not queue order), update a narrative of what each result implies, mutate survivors, mine failures, steal cross-domain ideas, re-baseline after every win, and stop only when the narrative stops gaining new entries.

You do not improvise the loop. You dispatch every step to the right `re-*` skill. You do not ask the user questions except (a) one metric clarifier at engine start if the metric is genuinely ambiguous, (b) one batched cost-ceiling question per round when paid-remote spend would exceed the ceiling, and (c) one target-hit confirmation if the user-stated target is reached. Anything else, you decide and act.

## Persona

Patient, disciplined, narrative-driven. You treat the narrative document as your working memory; you do not trust the leaderboard alone. You spend compute when it produces information; you do not spend compute when it confirms what you already know. You favor the experiment whose result would change the narrative most, regardless of expected sign. You are honest about negatives — they are knowledge.

## The skills

| Skill | When |
|---|---|
| `re-frame-problem` | First action; also re-invoked by `re-zoom-out`. Writes `dossier.md`. |
| `re-mine-literature` | At engine start (after framing) and every K=3 iterations. Narrative-driven literature pass. |
| `re-generate-hypotheses` | After every `re-mine-literature` and after every `re-update-narrative`. Five-source generator with non-zero quotas. |
| `re-select-next` | After every `re-generate-hypotheses`. Picks the highest-expected-information-gain × cost⁻¹ experiment. |
| `re-update-narrative` | After every iteration. Forced fields prevent score-only updates. |
| `re-detect-plateau` | After every `re-update-narrative`. Returns continue / continue-but-diversify / zoom-out / stop-and-write. |
| `re-zoom-out` | Only when `re-detect-plateau` returned `zoom-out`. Local-optima escape. |
| `re-write-up` | When `re-detect-plateau` returned `stop-and-write`, when the user interrupts, or when target is hit and user chose to stop. |

The engine also dispatches each iteration's experiment to one of the existing domain sub-agents — `ml-engineer` (tabular), `cv-engineer`, `nlp-engineer`, `llm-engineer`. The sub-agent runs its existing `research → decide → plan → write → execute → verify → debug → review` loop end-to-end, producing artifacts in the iteration directory.

## The outer loop

```
1. re-frame-problem      → dossier.md, narrative.md skeleton, status.json
2. re-mine-literature    → reading/<ts>.md, seed claims into narrative
3. re-generate-hypotheses → ~20 hypotheses across five sources (initial seed)
4. re-select-next        → pick the next experiment + domain route
5. dispatch to <domain sub-agent> with hypothesis + champion + iteration_dir
6. re-update-narrative   → narrative_delta.md, leaderboard append, re-rank
7. re-detect-plateau     → continue | continue-but-diversify | zoom-out | stop-and-write

Branch on the plateau decision:
- continue              → goto 4
- continue-but-diversify → goto 3 (with extra adversarial weight)
- zoom-out              → re-zoom-out, then goto 3 (initial-seed mode under new framing)
- stop-and-write        → re-write-up, then exit

Every K=3 iterations, also goto 2 (background literature pass) before goto 4.
```

## First message to user

When you start, print exactly:

```
Engine started. Working on: <one-line problem statement from dossier>.
Champion: <baseline metric>. Iteration 1 selected: <hypothesis one-line>.
Workdir: <abs path to research_engine/>

I'll keep working. Interrupt at any time. I'll surface a question only if I need
to spend on a paid remote past the per-session ceiling, if the user-stated
target is reached, or if the metric is genuinely ambiguous at framing time.
```

After every iteration, print one line:

```
Iter <NNN>: <hypothesis one-line> → <metric value> (<delta vs champion>). <plateau decision>.
```

That is the only ongoing user-facing output until report-time. No verbose narration, no per-skill commentary. The user reads the workdir if they want detail.

## When to ask the user (the only three permitted questions)

1. **At framing time, if the metric is undecidable from the request.** One multiple-choice question; then proceed silently.
2. **At target-hit time.** One question: continue past target or stop?
3. **When projected paid-remote spend over the next round of selected experiments would exceed `cost_ceiling_usd`.** One batched question per round, never per experiment. Local CPU/MPS/GPU compute is never gated this way.

You do NOT ask the user for plan approval, hypothesis approval, or experiment approval. The user contract is: "I want you to work on this; you may interrupt me only for the three permitted questions; otherwise act."

## Resume

If the user starts a session in a directory with an existing `<workdir>/research_engine/`, read `status.json`. If `state ∈ {running, awaiting_user, paused}`, resume from `next_action`. Print:

```
Resuming engine from iter <NNN>. Champion: <metric>. Last action: <last_event_kind>.
Next: <next_action>.
```

If `status.json` is inconsistent (e.g., `next_action` references an iteration directory that doesn't exist), do NOT auto-recover. Print the inconsistency and ask the user to confirm the recovery path.

## Hard constraints

- Never invoke a `re-*` skill outside this agent. They are engine-only.
- Never bypass the outer loop. Every iteration goes `select → dispatch → update → plateau-check`.
- Never spend on paid remotes past the cost ceiling without surfacing the batched question.
- Never delete narrative content. Append-only discipline.
- Never claim completion without `re-write-up` having run.
- The first message and per-iteration line are the ONLY user-facing outputs during the loop. No "thinking out loud" narration.
````

- [ ] **Step 2: Verify required sections present**

Run: `grep -c "^## " ML_Engineer/agents/research-engine.md`
Expected: at least `5`.

- [ ] **Step 3: Verify the agent's description discriminator is sharp**

Run: `grep "Do NOT use for one-shot questions" ML_Engineer/agents/research-engine.md`
Expected: the line is present.

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/agents/research-engine.md
git commit -m "v0.3.0: add research-engine agent — owns the continuous outer loop, dispatches inner steps to existing domain sub-agents"
```

---

### Task 12: Router prologue — `ml-engineer.md` (Step 0)

Add the engine-vs-one-shot disambiguator before the existing domain routing.

**Files:**
- Modify: `ML_Engineer/agents/ml-engineer.md` (insert Step 0 before existing Step 1)

- [ ] **Step 1: Read the current router prologue**

Run: `head -60 ML_Engineer/agents/ml-engineer.md`

Identify the line that says `## Routing rules`. The new Step 0 will be inserted *before* the existing rules, and the existing rules (currently `1. Strong signal — direct dispatch.` etc.) will become *Step 1*, *Step 2*, *Step 3*. The numbering changes; the content does not.

- [ ] **Step 2: Insert the Step 0 block**

Edit `ML_Engineer/agents/ml-engineer.md`. Find the section heading:

```markdown
## Routing rules

Apply these in order; first match wins:

1. **Strong signal — direct dispatch.**
```

Replace with:

```markdown
## Routing rules

Apply these in order; first match wins.

### Step 0 — Engine vs. one-shot

Before any domain routing, decide whether this is a problem-shaped request (engine) or a question-shaped request (one-shot transactional path).

- **Problem-shaped** — the user describes a metric to push past, a baseline to beat, an optimization goal, an open question with no obvious next step, or anywhere there is a frontier the user hasn't crossed. Phrasing signals: "beat", "improve", "push past", "find the best", "optimize", "we're stuck at", "current approach is X, want better". Dispatch to the `research-engine` agent. The engine internally routes candidate experiments to the appropriate domain sub-agent — you do not need to figure out the domain here.
- **Question-shaped** — single fact lookup, recipe question, single-answer scope. Phrasing signals: "what's the right LR for X", "how do I configure Y", "show me how to Z", "explain how X works". Proceed to Step 1 (existing domain routing).
- **Ambiguous** — ask exactly one disambiguator: *"Is this a one-shot question or a problem you want me to keep working on?"* — then route based on the answer. Do NOT guess.

The discriminator is **frontier vs. answer**, not domain. A frontier-shaped request in tabular ML routes to the engine; a recipe question in LLM finetuning routes through Step 1.

### Step 1 — Domain routing for one-shot tasks

If Step 0 returned "one-shot", apply the existing domain rules below.

1. **Strong signal — direct dispatch.**
```

(Keep the rest of the file unchanged. The existing `1. Strong signal — direct dispatch.`, `2. Ambiguous signal — ask one clarifying question.`, `3. Multi-domain task — stay in charge as router.` stay as-is, now nested under Step 1.)

- [ ] **Step 3: Verify Step 0 was inserted correctly**

Run: `grep -c "Step 0 — Engine vs" ML_Engineer/agents/ml-engineer.md && grep -c "Step 1 — Domain routing" ML_Engineer/agents/ml-engineer.md`
Expected: both counts = 1.

- [ ] **Step 4: Verify the existing rules are still present**

Run: `grep -c "Strong signal — direct dispatch" ML_Engineer/agents/ml-engineer.md`
Expected: ≥ 1.

- [ ] **Step 5: Commit**

```bash
git add ML_Engineer/agents/ml-engineer.md
git commit -m "v0.3.0: ml-engineer router — add Step 0 (engine vs. one-shot) before existing domain routing"
```

---

### Task 13: Router prologue — `cv-engineer.md`

Same Step 0 prologue. Sub-agents that get dispatched to from `ml-engineer` Step 1 also need to recognize problem-shaped requests within their own dispatch (a user might come straight to `cv-engineer` via the marketplace's agent picker without going through the router).

**Files:**
- Modify: `ML_Engineer/agents/cv-engineer.md`

- [ ] **Step 1: Read the current cv-engineer file structure**

Run: `head -40 ML_Engineer/agents/cv-engineer.md`

Identify the start of the body (after the frontmatter and persona — typically the first `## ...` heading, often `## The skills` or `## The loop`).

- [ ] **Step 2: Insert the Step 0 prologue immediately after the persona section**

Edit `ML_Engineer/agents/cv-engineer.md`. Find the first `## ...` heading after the persona block (the `## Persona` section, if present, otherwise the first content `##` heading) and insert *before* it:

```markdown
## Step 0 — Engine vs. one-shot (recap from router)

Before applying the loop below, decide whether this is a problem-shaped request (engine) or a question-shaped request (one-shot transactional path).

- **Problem-shaped** — frontier to push past, baseline to beat, optimization goal. Dispatch to the `research-engine` agent. The engine will route iteration experiments back to you for execution; do not run the loop directly.
- **Question-shaped** — single fact, recipe, configuration question. Proceed with the loop below.
- **Ambiguous** — ask exactly one disambiguator: *"Is this a one-shot question or a problem you want me to keep working on?"*

This recap exists so direct dispatches to this agent (bypassing the top-level router) get the same Step 0 treatment.

```

- [ ] **Step 3: Verify the insertion**

Run: `grep -c "Step 0 — Engine vs" ML_Engineer/agents/cv-engineer.md`
Expected: count = 1.

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/agents/cv-engineer.md
git commit -m "v0.3.0: cv-engineer — add Step 0 prologue so direct dispatches still get engine vs. one-shot routing"
```

---

### Task 14: Router prologue — `nlp-engineer.md`

Same insertion as Task 13.

**Files:**
- Modify: `ML_Engineer/agents/nlp-engineer.md`

- [ ] **Step 1: Read the current nlp-engineer file structure**

Run: `head -40 ML_Engineer/agents/nlp-engineer.md`

- [ ] **Step 2: Insert the Step 0 prologue**

Edit `ML_Engineer/agents/nlp-engineer.md`. Find the first content `##` heading after the persona block and insert *before* it:

```markdown
## Step 0 — Engine vs. one-shot (recap from router)

Before applying the loop below, decide whether this is a problem-shaped request (engine) or a question-shaped request (one-shot transactional path).

- **Problem-shaped** — frontier to push past, baseline to beat, optimization goal. Dispatch to the `research-engine` agent. The engine will route iteration experiments back to you for execution; do not run the loop directly.
- **Question-shaped** — single fact, recipe, configuration question. Proceed with the loop below.
- **Ambiguous** — ask exactly one disambiguator: *"Is this a one-shot question or a problem you want me to keep working on?"*

This recap exists so direct dispatches to this agent (bypassing the top-level router) get the same Step 0 treatment.

```

- [ ] **Step 3: Verify**

Run: `grep -c "Step 0 — Engine vs" ML_Engineer/agents/nlp-engineer.md`
Expected: count = 1.

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/agents/nlp-engineer.md
git commit -m "v0.3.0: nlp-engineer — add Step 0 prologue so direct dispatches still get engine vs. one-shot routing"
```

---

### Task 15: Router prologue — `llm-engineer.md`

Same insertion as Tasks 13/14.

**Files:**
- Modify: `ML_Engineer/agents/llm-engineer.md`

- [ ] **Step 1: Read the current llm-engineer file structure**

Run: `head -40 ML_Engineer/agents/llm-engineer.md`

- [ ] **Step 2: Insert the Step 0 prologue**

Edit `ML_Engineer/agents/llm-engineer.md`. Find the first content `##` heading after the persona block and insert *before* it:

```markdown
## Step 0 — Engine vs. one-shot (recap from router)

Before applying the loop below, decide whether this is a problem-shaped request (engine) or a question-shaped request (one-shot transactional path).

- **Problem-shaped** — frontier to push past, baseline to beat, optimization goal. Dispatch to the `research-engine` agent. The engine will route iteration experiments back to you for execution; do not run the loop directly.
- **Question-shaped** — single fact, recipe, configuration question. Proceed with the loop below.
- **Ambiguous** — ask exactly one disambiguator: *"Is this a one-shot question or a problem you want me to keep working on?"*

This recap exists so direct dispatches to this agent (bypassing the top-level router) get the same Step 0 treatment.

```

- [ ] **Step 3: Verify**

Run: `grep -c "Step 0 — Engine vs" ML_Engineer/agents/llm-engineer.md`
Expected: count = 1.

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/agents/llm-engineer.md
git commit -m "v0.3.0: llm-engineer — add Step 0 prologue so direct dispatches still get engine vs. one-shot routing"
```

---

### Task 16: Document v0.3.0 in README

Add the research engine to the plugin README.

**Files:**
- Modify: `ML_Engineer/README.md`

- [ ] **Step 1: Read the existing v0.2.0 section**

Run: `grep -n "## Deep learning support\|## v0.2.0\|## Inspiration" ML_Engineer/README.md | head -20`

Identify the line numbers of the `## Deep learning support (v0.2.0 — full v1 release)` section header and the next top-level `## ` after it.

- [ ] **Step 2: Insert the v0.3.0 section after the v0.2.0 section**

Edit `ML_Engineer/README.md`. Find the line `## Inspiration & credit` (or whatever the section after v0.2.0 is). Insert immediately *before* it:

```markdown
## Research engine (v0.3.0 — autonomous problem-solver)

The plugin now includes a continuous research engine. For any problem-shaped request — a metric to push past, a baseline to beat, an optimization goal, an open question with no obvious next step — the router silently dispatches to the engine instead of running a one-shot transactional task.

The engine works the way a rigorous human data scientist works: read continuously, maintain a living hypothesis list across five generation sources (literature, mutation of survivors, failure-mining, cross-domain analogy, adversarial wild-card), select the highest-expected-information experiment each round, dispatch it to the appropriate domain sub-agent for execution, update a narrative document (the engine's working memory), re-baseline after every win, and stop only when the narrative stops gaining new entries — not when a counter hits zero.

```
1. re-frame-problem      → dossier.md, narrative.md skeleton, status.json
2. re-mine-literature    → reading/<ts>.md, seed claims into narrative
3. re-generate-hypotheses → ~20 hypotheses across five sources
4. re-select-next        → pick highest-info-gain × cost⁻¹ experiment + domain route
5. dispatch to <ml-engineer | cv-engineer | nlp-engineer | llm-engineer>
6. re-update-narrative   → forced-field updates; re-rank; re-baseline if won
7. re-detect-plateau     → continue | continue-but-diversify | zoom-out | stop-and-write
   - zoom-out → re-zoom-out → goto 3 (initial-seed under new framing)
   - stop-and-write → re-write-up → exit
```

**Eight engine-only skills** (`re-*`) implement the loop. They only fire from inside the `research-engine` agent — users do not invoke them manually.

| Skill | Role |
|---|---|
| `re-frame-problem` | Builds the problem dossier (metric, baseline, data shape, prior knowledge, stop criteria) |
| `re-mine-literature` | Narrative-driven literature reader; bounded budget (4 queries / 6 fetches per pass) |
| `re-generate-hypotheses` | Five-source generator with non-zero adversarial / cross-domain quotas |
| `re-select-next` | Picks next experiment by expected information gain × cost⁻¹; diversity tie-break |
| `re-update-narrative` | Forced-field discipline (ruled-out + suspected required) prevents score-only updates |
| `re-detect-plateau` | Reads narrative deltas (NOT the metric); decides continue / zoom-out / stop |
| `re-zoom-out` | Local-optima escape via metric / unit / decomposition / data-slice shift; never deletes narrative |
| `re-write-up` | Final report with mandatory "what didn't work" + "honest assessment" sections |

**The engine asks the user only three questions during a session:** (1) at framing time, if the metric is undecidable from the request; (2) at target hit, continue past target or stop; (3) when projected paid-remote spend over the next round would exceed the cost ceiling (default $5/session; configurable). Local CPU/MPS/GPU compute is never gated. Anything else, the engine decides and acts.

**Workdir layout** under `./newton_workdir/<UTC-timestamp>/research_engine/`:

```
research_engine/
├── dossier.md                  # problem framing (re-frame-problem)
├── narrative.md                # the engine's working memory — append-only sections
├── hypotheses.jsonl            # live hypothesis list (versioned records)
├── hypotheses_archive.jsonl    # killed / superseded hypotheses
├── leaderboard.jsonl           # one record per executed experiment
├── status.json                 # current engine state (resumable)
├── champion/                   # symlink to current best run
├── iterations/<NNN>/           # one directory per iteration with the sub-agent's standard outputs
└── reading/<ts>.md             # one file per literature pass
```

The workdir survives across Claude Code sessions. Resume is a first-class operation: open a new session in the same directory, the engine reads `status.json` and continues.

See [`docs/superpowers/specs/2026-05-07-research-engine-design.md`](docs/superpowers/specs/2026-05-07-research-engine-design.md) for the full spec, [`docs/superpowers/specs/research-engine-workdir-schema.md`](docs/superpowers/specs/research-engine-workdir-schema.md) for the workdir schema, and [`docs/superpowers/plans/2026-05-08-research-engine-implementation.md`](docs/superpowers/plans/2026-05-08-research-engine-implementation.md) for the implementation plan.

```

- [ ] **Step 3: Verify the insertion**

Run: `grep -c "Research engine (v0.3.0" ML_Engineer/README.md && grep -c "re-frame-problem" ML_Engineer/README.md`
Expected: both counts ≥ 1.

- [ ] **Step 4: Commit**

```bash
git add ML_Engineer/README.md
git commit -m "v0.3.0: README — document the research engine (eight re-* skills + the research-engine agent)"
```

---

### Task 17: Acceptance dogfood

End-to-end check that the engine starts on a problem-shaped request, the disambiguator works on a one-shot request, and resume works.

**Files:**
- Read-only: all of the above.

- [ ] **Step 1: Confirm all eight re-* skill files exist with valid frontmatter**

Run:

```bash
for s in re-frame-problem re-mine-literature re-generate-hypotheses re-select-next re-update-narrative re-detect-plateau re-zoom-out re-write-up; do
  test -f "ML_Engineer/skills/$s/SKILL.md" && echo "OK: $s" || echo "MISSING: $s"
done
```

Expected: 8 lines, all `OK`.

- [ ] **Step 2: Confirm the research-engine agent exists and routes correctly**

Run: `test -f ML_Engineer/agents/research-engine.md && echo "OK"`
Expected: `OK`.

Run: `grep -c "Step 0 — Engine vs" ML_Engineer/agents/ml-engineer.md ML_Engineer/agents/cv-engineer.md ML_Engineer/agents/nlp-engineer.md ML_Engineer/agents/llm-engineer.md`
Expected: each agent file shows count = 1.

- [ ] **Step 3: Confirm the workdir schema doc is referenced from every engine skill**

Run: `grep -l "research-engine-workdir-schema" ML_Engineer/skills/re-*/SKILL.md | wc -l`
Expected: at least 5 (workdir-schema is referenced by the skills that read/write workdir files; some skills may not need it).

- [ ] **Step 4: Confirm plugin.json is at 0.3.0**

Run: `python3 -c "import json; v=json.load(open('ML_Engineer/.claude-plugin/plugin.json'))['version']; print(v); assert v=='0.3.0'"`
Expected: `0.3.0` printed; no assertion error.

- [ ] **Step 5: Manual dogfood — engine vs. one-shot routing**

In a new Claude Code session in this directory, run two test prompts and verify behavior:

**Prompt A (problem-shaped, expected: engine starts):**

> "I'm trying to beat 0.87 AUC on this fraud-detection CSV. Current approach is gradient-boosted trees with default hyperparameters. Push it as far as you can."

Expected behavior:
- The router selects Step 0 → engine.
- The `research-engine` agent prints the "Engine started." message.
- A `<workdir>/research_engine/dossier.md` is written.
- An iteration begins.

**Prompt B (question-shaped, expected: one-shot path):**

> "What's a good learning rate for QLoRA on Llama-3 8B?"

Expected behavior:
- The router selects Step 0 → one-shot.
- The router proceeds to Step 1 (existing domain routing).
- `llm-engineer` answers the question without spinning up the engine.
- No `research_engine/` directory is created.

**Prompt C (ambiguous, expected: disambiguator):**

> "Help me with this dataset."

Expected behavior:
- The router asks: *"Is this a one-shot question or a problem you want me to keep working on?"*
- No action taken until the user replies.

If any of A/B/C fails the expected behavior, the corresponding agent's Step 0 prologue needs revision — fix and re-test.

- [ ] **Step 6: Manual dogfood — resume**

In the same Claude Code session as Prompt A, after at least one iteration completes, exit the session. Then:

1. Re-open Claude Code in the same directory.
2. Type: "continue".
3. Expected: the agent reads `status.json` and prints `Resuming engine from iter <NNN>. Champion: <metric>. Last action: <last_event_kind>. Next: <next_action>.` — then resumes.

If resume fails (e.g., `status.json` not found, agent restarts from scratch), the `research-engine.md` agent's resume section needs revision.

- [ ] **Step 7: Final commit**

If anything was fixed during Steps 5–6, commit those fixes:

```bash
git add -A ML_Engineer/agents/ ML_Engineer/skills/re-*/
git commit -m "v0.3.0: acceptance fixes from dogfooding — see commit body for details"
```

If nothing needed fixing, the dogfood passed; no commit is needed.

- [ ] **Step 8: Tag the v0.3.0 release**

```bash
git tag -a v0.3.0 -m "v0.3.0: research engine — autonomous problem-solver as the default execution mode

- New agent: research-engine
- 8 new engine-only skills: re-frame-problem, re-mine-literature, re-generate-hypotheses, re-select-next, re-update-narrative, re-detect-plateau, re-zoom-out, re-write-up
- Step 0 routing prologue in ml-engineer / cv-engineer / nlp-engineer / llm-engineer (engine vs. one-shot)
- Persistent workdir under <workdir>/research_engine/ with resume support
- Existing 47 skills unchanged — engine wraps them as primitives"
```

(Tag only — do not push unless the user asks.)

---

## Self-review

Spec coverage check:

- ✅ Problem dossier (`re-frame-problem`) — Task 3.
- ✅ Continuous reader driven by narrative (`re-mine-literature`) — Task 4.
- ✅ Five-source hypothesis generator with non-zero adversarial/cross-domain quotas — Task 5.
- ✅ Selector by expected information gain × cost⁻¹ with diversity tie-break — Task 6.
- ✅ Forced-field narrative update preventing score-only updates — Task 7.
- ✅ Plateau detection via narrative, not metric — Task 8.
- ✅ Zoom-out with four axes (metric / unit / decomposition / data-slice) and append-only narrative — Task 9.
- ✅ Final report with mandatory "what didn't" and "honest assessment" — Task 10.
- ✅ Engine agent with persona, skill table, outer loop, resume — Task 11.
- ✅ Step 0 routing in all four agents — Tasks 12-15.
- ✅ Cost ceiling check before paid-remote spend — Task 6 (Step 3 cost ceiling) + Task 11 (engine "when to ask user" rule 3).
- ✅ Workdir schema as single source of truth — Task 2.
- ✅ Resume from `status.json` — Task 11 ("Resume" section).
- ✅ Three permitted user questions only — Task 11.
- ✅ Plugin version bumped — Task 1.
- ✅ README documented — Task 16.
- ✅ Acceptance dogfood — Task 17.

Placeholder scan: no "TBD" / "TODO" / "implement later" / "fill in details" anywhere in the plan. Every step has the exact content the engineer needs.

Type consistency: all skill names match the spec (`re-frame-problem`, `re-mine-literature`, `re-generate-hypotheses`, `re-select-next`, `re-update-narrative`, `re-detect-plateau`, `re-zoom-out`, `re-write-up`). All workdir paths match the schema doc (`dossier.md`, `narrative.md`, `hypotheses.jsonl`, `hypotheses_archive.jsonl`, `leaderboard.jsonl`, `status.json`, `champion/`, `iterations/<NNN>/`, `reading/<ts>.md`). All status states (`initializing`, `running`, `awaiting_user`, `paused`, `stopped`) are referenced consistently.
