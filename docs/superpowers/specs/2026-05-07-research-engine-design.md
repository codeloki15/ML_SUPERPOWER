# Research Engine — Design Spec

**Date:** 2026-05-07
**Plugin:** `ml-engineer`
**Version target:** `0.3.0`
**Status:** Draft, pending user review

## Problem

The plugin today is **transactional**. The `ml-engineer` agent and its sub-agents (`cv-engineer`, `nlp-engineer`, `llm-engineer`) run a one-shot loop per task: research → decide → plan → write → execute → verify → debug → review. The loop terminates when the task's queue empties.

A rigorous human data scientist does not work that way. They keep a problem open, read continuously, maintain a living hypothesis list, run experiments selected by *expected information gain* rather than queue order, write a narrative of what each result implies, mutate survivors, mine failures, steal cross-domain ideas, re-baseline after every win, and only stop when the narrative stops gaining new entries — not when a counter hits zero.

The plugin currently has no equivalent. The 47 existing skills are the right *primitives* but no controller stitches them into a continuing search. The user has to play the controller themselves: re-invoke skills, track progress in their head, decide what to try next. That is exactly the work the system should be doing.

## Goal

Ship a **continuous research engine** as the default execution mode for every problem-shaped request. The user describes a problem; the engine works on it until the problem is solved, the gain plateaus on the *narrative* (not just the metric), or the user redirects. The existing 47 skills become the engine's primitives. The transactional one-shot mode is preserved only for genuine one-shot questions ("what's a good LR for Llama 3 LoRA") — everything that has a frontier to push past goes through the engine.

The engine's discipline is the **narrative document**, not the leaderboard. A leaderboard plateaus and tells you nothing about why; a narrative tells you what's been ruled out, what's now suspected, and what experiment would change that. The selector reads the narrative; the hypothesis generators read the narrative; the stopping rule reads the narrative.

## Non-goals (v1)

- Multi-tenancy or sharing engine state between users.
- Distributed engine itself (the engine is single-process; *experiments it dispatches* can run distributed via existing `dl-distributed` / `dl-remote-execute` skills).
- Visual dashboards / web UI. State is files in the workdir; views are CLI commands.
- Replacing the existing transactional skills. They keep working as-is and become the engine's primitives.
- Fully unattended *autonomous compute spend* across paid remotes without ever surfacing cost. Spend that exceeds a per-engine ceiling surfaces to the user. Within the ceiling, the engine spends without asking — that is the explicit user contract for engine mode.
- A new domain (audio, graphs, time-series DL, RL) — out of scope; the engine only orchestrates over what the existing skills already cover.
- A novel optimizer for individual experiments. The engine does not invent new training algorithms; it composes what the skills produce.

## How a human data scientist actually works (the model we're encoding)

Five things happen continuously, none on a fixed schedule. The engine's design mirrors all five.

1. **Read.** Drive literature mining from *what the current frontier doesn't explain*, not a one-shot search. Each round reads something new, biased toward whatever the narrative most recently flagged as unknown.
2. **Hypothesize.** Maintain a living, themed list. Every observation — successful run, failed run, new paper paragraph, user comment — produces, kills, mutates, or re-ranks entries. The list is never "done."
3. **Run the most informative thing right now.** Not the next item in a queue. The selector picks the experiment whose result would *change the narrative most*, regardless of expected sign. Negative results are knowledge.
4. **Update the narrative.** After every run, write in plain language what was learned, what is now ruled out, what is now suspected, what hypothesis class moved or didn't. The narrative is the memory.
5. **Re-baseline and decide whether to continue.** Stop when the narrative stops gaining new entries over the last K experiments — not when the metric plateaus alone. Metric plateau without narrative plateau means stuck-in-local-well; the system has more to try.

## Architecture

One new agent, one new directory of engine-only skills, one persistent workdir layout. The existing 47 skills are unchanged.

```
ML_Engineer/
├── .claude-plugin/
│   ├── plugin.json                 (bumped to 0.3.0)
│   └── marketplace.json
├── agents/
│   ├── ml-engineer.md              (router prologue updated — see Routing)
│   ├── cv-engineer.md              (engine-aware — see Sub-agent integration)
│   ├── nlp-engineer.md             (engine-aware)
│   ├── llm-engineer.md             (engine-aware)
│   └── research-engine.md          (NEW — the continuous controller)
├── skills/
│   ├── ml-engineer-*               (existing 15 — unchanged)
│   ├── dl-*                        (existing 33 — unchanged)
│   └── re-*                        (NEW — 8 engine-only skills, see below)
├── docs/superpowers/
│   ├── specs/
│   │   └── 2026-05-07-research-engine-design.md   (this file)
│   └── plans/
│       └── 2026-05-08-research-engine-implementation.md   (forthcoming)
└── README.md                       (updated)
```

### The engine agent

`agents/research-engine.md` owns the continuous loop. It is dispatched to by the router (see Routing) and by each domain sub-agent when the task is problem-shaped rather than question-shaped. Once dispatched, it owns the session until the user interrupts or it self-stops.

The engine's inner step is the **existing transactional loop** (`research → decide → plan → write → execute → verify → debug → review`), invoked through the appropriate domain sub-agent for the candidate being run. The engine does not reimplement that loop — it wraps it.

The engine's outer step is the **continuous loop**:

```
loop forever:
  read       (re-* literature + residuals + the narrative itself)
  hypothesize (mutate, recombine, mine failures, cross-domain analogies)
  select     (pick the highest-expected-information experiment)
  run        (dispatch through the appropriate sub-agent's transactional loop)
  update     (rewrite the narrative; re-rank hypotheses; re-baseline if won)
  decide     (continue / zoom out / stop, based on the narrative)
```

Critically, the five sub-actions of `read / hypothesize / select / run / update` are **not** strictly sequential. `read` runs in the background between rounds. `hypothesize` fires on every event — new paper, finished run, user comment. `select` runs as soon as compute is free. The engine is event-driven, not phase-locked.

### The eight new skills

All in `skills/re-*` (`re` = research-engine). Engine-only — none are useful outside the engine, so they have sharp triggers that only fire from `research-engine.md`.

| Skill | Role |
|---|---|
| `re-frame-problem` | Builds the **problem dossier** at engine start. Captures metric, baseline, data shape, what's already known to work, what's already known to fail, the user's stop-criteria preferences. Writes `dossier.md`. Re-invoked when the user redirects. |
| `re-mine-literature` | Continuous reader. Pulls arxiv / Kaggle / HF cookbook / conference proceedings. Driven by *what the narrative most recently flagged as unknown*, not a fixed query. Writes findings into the narrative as **claims**, not citations. Wraps `dl-prior-art` and `ml-engineer-research` but reads the narrative first to bias the search. |
| `re-generate-hypotheses` | The five-source generator. **(a)** literature pull, **(b)** mutation of survivors, **(c)** failure mining (counter-hypotheses from the bottom half), **(d)** cross-domain analogy (vision ↔ NLP ↔ tabular ↔ chemistry — *forced* analogies), **(e)** adversarial/red-team ("the dumbest thing that might work"). All five fire every round. The (e) budget is non-negotiable: at least one wild-card per round, regardless of belief. Writes hypotheses into the live list with theme + lineage (which source, which parent if mutated). |
| `re-select-next` | The brain of the engine. Scores every hypothesis on the live list by **expected information gain × cost⁻¹**. Information gain ≈ *how much would this result, regardless of sign, change what the narrative now claims*. Tie-breaks toward diversity (don't run three close cousins consecutively). Returns the next experiment to dispatch. |
| `re-update-narrative` | After every run, updates `narrative.md` in plain language. Says: what was tested, what the result was, what the result implies, what is now ruled out, what is now newly suspected. *Forced* fields prevent score-only updates. Re-ranks the hypothesis list given the new claim set. Detects when a champion has been beaten and rebaselines. |
| `re-detect-plateau` | Reads the narrative, not the metric. Plateau = the last K updates added no new ruled-out claims and no new suspected claims. K is adaptive (smaller for fast cheap experiments, larger for expensive ones). Returns one of: `continue`, `zoom-out` (re-frame the problem; the engine is exploring the wrong space), `stop-and-write` (no new learning available). |
| `re-zoom-out` | Invoked when `re-detect-plateau` returns `zoom-out`. Re-runs `re-frame-problem` with explicit instruction to **change the framing** (different metric? different unit of analysis? different problem decomposition?). Often produces a new dossier under which the existing hypothesis list is partly invalidated, partly re-prioritized. The plug-and-pray escape from local optima. |
| `re-write-up` | Produces the final report when the engine self-stops. Includes: what was tried, what worked, what didn't, *why* (from the narrative), the new champion, recommended next steps if the user wants to continue manually. Saves alongside the workdir. |

### Workdir layout (persistent, per-engine-session)

The engine writes a structured durable directory the user can inspect or re-attach to:

```
./newton_workdir/<UTC-timestamp>/research_engine/
├── dossier.md                  # the problem framing, baseline, stop-criteria
├── narrative.md                # the running scientific narrative — append-only sections
├── hypotheses.jsonl            # the live list (each line a versioned hypothesis record)
├── hypotheses_archive.jsonl    # killed / superseded hypotheses, with reason
├── leaderboard.jsonl           # one record per executed experiment, with metric + cost
├── champion/                   # symlink to the current best run's artifacts
├── iterations/
│   ├── 001/                    # one directory per executed experiment
│   │   ├── hypothesis.json
│   │   ├── plan.md
│   │   ├── step_*.py           # standard ml-engineer-write-code outputs
│   │   ├── verify_*.py
│   │   ├── results.json
│   │   └── narrative_delta.md  # what this run added/removed from the narrative
│   ├── 002/
│   └── ...
├── reading/
│   ├── 2026-05-07T14-22Z.md    # one file per re-mine-literature pass
│   └── ...
└── status.json                 # current state: running / paused / stopped + last-event
```

The workdir survives across Claude Code sessions. Re-attaching is a first-class operation: the user opens a new session, says "continue the engine in `./newton_workdir/<ts>/research_engine`", and the engine resumes from `status.json` + the narrative.

### Routing

The router prologue in `agents/ml-engineer.md` gets a new top-level decision *before* the existing domain routing.

```
Step 0 — Engine vs. one-shot.
  - If the request describes a PROBLEM (a metric to beat, a frontier to push, an
    optimization goal, an open question with no obvious next step): dispatch to
    `research-engine` agent. The engine will internally route candidate experiments
    to the appropriate domain sub-agent.
  - If the request is a QUESTION (single fact, single recipe, "what's a good X for Y",
    "how do I configure Z"): proceed with existing transactional routing.
  - Ambiguous: ask exactly one disambiguator: "Is this a one-shot question or a
    problem you want me to keep working on?"

Step 1 (existing) — Domain routing for one-shot tasks.
  ... (existing rules, unchanged)
```

The **discriminator**: a *problem* has a baseline-or-target, a metric, and a frontier the user hasn't crossed. A *question* has an answer that fits in one response.

Sub-agents (`cv-engineer`, `nlp-engineer`, `llm-engineer`) get the same Step 0 prologue. So a user who says "finetune Qwen on this data, beat 0.62 BLEU" routes to `llm-engineer` → engine. A user who says "what chat template does Qwen use" routes to `llm-engineer` → one-shot.

### Sub-agent integration

When the engine selects an experiment, it dispatches to the appropriate domain sub-agent (`cv-engineer` / `nlp-engineer` / `llm-engineer` / tabular `ml-engineer`) with three pieces of context:

1. The hypothesis (one-line claim + concrete change to test)
2. The current champion (so the sub-agent knows what to compare against)
3. The iteration directory path (so all sub-agent outputs land there)

The sub-agent runs its existing transactional loop end-to-end, producing the standard outputs in the iteration directory. On completion, control returns to the engine, which invokes `re-update-narrative`.

The sub-agent does not need to know it's running inside the engine — it just sees a task with a hypothesis, a baseline, and a destination directory. This keeps the existing skills' contracts intact.

### The narrative document — non-trivial detail

The narrative is the engine's most important artifact. Its structure is fixed so future iterations can reliably parse it:

```markdown
# Narrative — <problem name>
**Started:** <ts>   **Last updated:** <ts>   **Champion:** <metric value>

## Ruled out
- <claim> (iter <n>: <one-line reason>)
- ...

## Currently suspected
- <claim> (iter <n>: <one-line reason>) [confidence: low|med|high]
- ...

## Open questions
- <question> (raised iter <n>: <one-line reason>)
- ...

## Per-iteration log
### Iter 001 — <hypothesis one-line>
**Result:** <metric, ± vs champion>
**Implies:** <one-line takeaway>
**Narrative delta:** added [...], removed [...], promoted [...] from suspected to ruled-out.
...
```

The four top sections are **the engine's working memory**. Hypothesis generation, selection, plateau detection all read these sections (not the per-iteration log) to make their decisions.

### Stopping (the engine's call, not a counter)

The engine self-stops in three cases, in order:

1. **Target hit.** The user-stated target metric is reached. (Engine asks once: "Continue past the target, or stop?" — the only proactive question after engine start.)
2. **Narrative plateau.** `re-detect-plateau` returns `stop-and-write`. The engine has tried the obvious moves and the unobvious moves and exhausted its hypothesis-generation sources without producing a new claim for K rounds.
3. **User interrupt.** Standard Claude Code interrupt. Engine writes a final narrative state and stops cleanly.

A *metric plateau without narrative plateau* triggers `re-zoom-out` instead of stopping. This is the local-optimum escape.

### Cost surfacing (the one place autonomy yields)

The engine has compute autonomy by default — that is the explicit user contract for engine mode. But unbounded paid-remote spend is not. Concretely:

- The engine maintains a running per-session spend estimate (`leaderboard.jsonl` includes per-iteration cost from `dl-remote-execute`).
- When projected spend over the next round of selected experiments would exceed a configurable ceiling (default $5; user can raise via `--engine-budget` or set to `unlimited`), the engine surfaces a single batched "About to spend $X on N experiments — proceed?" question.
- Local CPU/MPS/GPU compute is never gated this way. Only paid remotes.

This is the *only* proactive interruption pattern after engine start.

### Auto-trigger — the user shouldn't have to know it exists

The user types problem-shaped requests. The router silently dispatches to the engine. The engine starts working. The user sees:

```
Engine started. Working on: <one-line problem statement>.
Champion: <baseline metric>. Iteration 1 selected: <hypothesis one-line>.
Workdir: ./newton_workdir/<ts>/research_engine/

I'll keep working. Interrupt at any time. I'll surface a question only if I need
to spend on a paid remote past the per-session ceiling.
```

The user does not invoke `/research-engine` or any `re-*` skill manually. The engine and its skills only fire from inside the engine agent.

## Components

The eight new skills, the new agent, the routing changes, and the workdir layout. Each new skill follows the existing `SKILL.md` conventions (frontmatter with `name`, `description`, `license`, `metadata`; trigger language in `description`; sharp anti-triggers).

## Data flow

1. User describes a problem.
2. Router (`ml-engineer.md`) dispatches to `research-engine`.
3. `research-engine` invokes `re-frame-problem` → writes `dossier.md`.
4. `re-mine-literature` runs an initial pass → writes `reading/<ts>.md` and seeds claims into the narrative's "Currently suspected" / "Open questions" sections.
5. `re-generate-hypotheses` populates the initial live list (≈ 20 hypotheses across the five sources).
6. **Loop:**
   - `re-select-next` returns the highest-expected-information experiment.
   - `research-engine` dispatches it to the appropriate domain sub-agent with the iteration directory path.
   - Sub-agent runs `research → decide → plan → write → execute → verify → debug → review`.
   - On return, `re-update-narrative` rewrites `narrative.md`, re-ranks hypotheses, re-baselines if there's a new champion.
   - `re-detect-plateau` decides: continue / zoom-out / stop-and-write.
   - In the background between iterations, `re-mine-literature` runs another reading pass biased by the latest narrative state; `re-generate-hypotheses` adds new candidates.
7. On stop, `re-write-up` produces the final report.

## Error handling

- **Sub-agent transactional loop fails** (e.g., `dl-debug-training` exhausts its 3-failure budget): the iteration is recorded as a *negative result* (still informative). `re-update-narrative` writes the failure mode as a new claim under "Currently suspected" or "Ruled out." The engine continues with the next-best hypothesis.
- **Literature mining produces no new claims for K rounds:** the literature search is exhausted relative to the current narrative; `re-generate-hypotheses` falls back harder on mutation, failure-mining, and adversarial sources.
- **Hypothesis generation produces only near-duplicates:** the engine forces a `re-zoom-out` rather than running redundant experiments. This is a strong plateau signal.
- **Workdir corruption / partial state on resume:** `status.json` is the source of truth. If absent or stale, the engine reconstructs from the last consistent iteration directory and a re-read of the narrative. No silent restart.
- **User interrupts mid-iteration:** the in-flight sub-agent task is allowed to complete (so the iteration directory is consistent), then the engine stops. If the user wants a hard kill, they kill the process.
- **Cost ceiling hit mid-round:** the engine surfaces the batched cost question, pauses, waits for user input. It does not silently demote to a smaller experiment without telling the user.

## Testing

Process testing only — these are skills + an agent, not a library. Validation is via dogfooding on real problems. The release gate is:

1. **Three problems with known good answers.** Run the engine on three benchmark problems where a strong reference solution is publicly known (e.g., a Kaggle competition with a known-winning playbook). Pass criteria: (a) the final metric is within 10% of the published winner's score on the same split, and (b) the narrative independently identifies at least three of the key moves the winner used (whether or not the metric matches exactly).
2. **One problem with no known good answer.** Run the engine on a novel problem the user brings. Validate by user judgment that the narrative is *coherent and informative* (not just a pile of runs), the hypotheses generated were diverse, and the stop point was reasonable.
3. **One ambiguous request.** Verify the router's "engine vs. one-shot" disambiguator fires correctly: a true one-shot question still gets a one-shot answer (no engine spin-up).
4. **One interrupt + resume.** Start an engine, interrupt mid-iteration, resume from the workdir. Verify the narrative is consistent and no work is lost or double-counted.
5. **One cost-ceiling event.** Configure a low ceiling, verify the batched cost question surfaces correctly and the engine respects the user's answer.

No new unit tests are added for the existing 47 skills — they are unchanged.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| The engine over-fires — turns one-shot questions into multi-hour searches | The router's Step 0 disambiguator. *Question vs. problem* is the discriminator, not domain. Sub-agents enforce the same check before dispatching to the engine. Anti-trigger explicitly excludes single-fact questions. |
| The narrative becomes a junk drawer (every claim gets added, none get removed) | `re-update-narrative` has a *forced* "ruled out" and "promoted to ruled out" field for every iteration. The skill prompt enforces deletion/promotion of stale claims when contradicted. Plateau detection treats no-removal-no-addition cycles as plateau signal — the engine self-corrects. |
| Hypothesis generation collapses to mutations of the current champion | The five-source generator has a *minimum quota per source per round*. Adversarial / cross-domain sources are non-zero by skill contract. Diversity is enforced in the selector. |
| Plateau detection too aggressive — engine stops while there's still signal | Plateau is computed from the narrative, not the metric. K is adaptive. The user can override the stop and tell the engine to continue from the current narrative state — the engine forks the hypothesis list with an instruction to widen the search. |
| Plateau detection too lax — engine runs forever burning compute | The stop signal is *narrative plateau*, which always fires eventually because the hypothesis space is finite and the narrative monotonically accumulates claims (until rebaselined by zoom-out). The cost ceiling is a hard backstop. |
| Engine state corrupted on resume | `status.json` + per-iteration directories are the source of truth. The engine refuses to resume from inconsistent state — it asks the user. No silent recovery that could double-count an iteration. |
| The engine and the existing transactional skills disagree about ownership of a task | Routing is single-pass and explicit. Step 0 picks engine *or* one-shot; once picked, the chosen path owns the task. No mid-task handoff back to the other path. The user can interrupt and re-route if they realize the wrong choice was made. |
| Auto-trigger surprises users — they wanted a one-shot answer and got an engine spinup | The engine's first message ("Engine started. Working on: ...") makes the mode explicit. The user can interrupt within seconds. The disambiguator covers genuine ambiguity. |
| Compute spend balloons | Cost ceiling default $5. Local compute unmetered. Paid-remote spend surfaces as a single batched question per round, not per experiment. |
| `re-zoom-out` is mis-triggered and discards real progress | Zoom-out does not delete the narrative — it appends a new framing under which existing claims are re-categorized. The original narrative is preserved. The user can revert by interrupt + manual edit. |
| Skill / agent count growth makes the plugin hard to reason about | The eight new skills are namespaced `re-*` and only fire from inside the engine agent. They are invisible to users who don't enter engine mode. The transactional surface is unchanged. |

## Open questions deferred to implementation

- Exact format of `hypotheses.jsonl` records (lineage tracking, theme tags). The plan will lock this.
- Whether `re-mine-literature` runs as a background subprocess between iterations or inline at the start of each round (depends on Claude Code's tool semantics for long-running background work).
- Default value of K in plateau detection (likely problem-class-dependent: smaller K for cheap-experiment regimes, larger for expensive).
- Whether to expose a CLI / status command for the user to inspect engine state mid-run, or rely on the user reading the workdir files. Defer.
- Ceiling currency / unit. Default `$5 per session`, but `dl-remote-execute` returns provider-native units; the engine needs a normalizer. The plan will spec this.
- Whether tabular `ml-engineer` (the router agent itself) needs to be split — it currently doubles as router and tabular orchestrator. With Step 0 added, it stays dual-purpose, but if the engine routing logic becomes too large, splitting the router into its own agent is a future cleanup.
