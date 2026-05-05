---
name: dl-prior-art
description: Use when the user describes a deep learning problem (image classification, segmentation, NER, LLM finetune, etc.) and is starting fresh OR at a fork in the road and wants to know what Kaggle competition winners and Hugging Face cookbook authors have done on similar problems. Returns a structured playbook (common patterns, diverging choices, recommended starting steps). Do NOT use for generic method-and-fact questions (use ml-engineer-research) or for problems already covered by an earlier dl-prior-art call in the same session.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Prior Art

Search the web for Kaggle competition winners and Hugging Face cookbook posts that solved problems similar to the user's, then synthesize a playbook of what worked. The output is structured guidance, not a citation list — internalize the evidence and pass actionable decisions to the orchestrator.

This skill complements `ml-engineer-research`. The two answer different question shapes:

- `ml-engineer-research` answers *"What is the current best practice for `<technique>` on `<data shape>`?"* — method-and-fact lookup.
- `dl-prior-art` answers *"What did the winners on similar problems actually do?"* — playbook lookup, including order of operations and negative results (things they tried and dropped).

## When to invoke

- User describes a new DL problem (CV / NLP / LLM / VLM) and wants a starting strategy.
- User is at an iteration fork (e.g., "should I distill or pseudo-label next?") and wants to know how winners on similar problems sequenced their work.
- User mentions a Kaggle competition, a HuggingFace cookbook task, or a published benchmark and wants the community's playbook.
- Useful before invoking domain-specific training skills (`dl-cv-classify`, `dl-nlp-token`, `dl-llm-instruction-tune`) so the orchestrator picks defaults that match what winners use.

## When NOT to invoke

- Generic method-and-fact questions ("what's the right LR for Llama 3 LoRA"). Use `ml-engineer-research`.
- The same problem class was researched earlier in the session — reuse the prior playbook.
- Trivial decisions (chart color, file name).
- The user has already specified their full pipeline and is asking for execution help, not strategy.

## Process

### Step 1 — Frame the problem as winner-shaped queries

Rewrite the user's problem into 2-3 search queries that target *winning solution writeups* and *cookbook recipes*. Include:

- The data domain (medical CT, satellite imagery, legal text, SQL generation, drug screening, etc.).
- The task type (classification, segmentation, detection, NER, instruction-tuning, etc.).
- The competition or cookbook qualifier: include `Kaggle winning solution`, `competition winner writeup`, `HuggingFace cookbook`, `Kaggle <competition_name> 1st place`, or `<dataset> SOTA approach`.
- A recency filter: prefer the last 2 years; append `2024 OR 2025 OR 2026` for fast-moving topics (LLM finetuning, VLMs).

**Bad query:** `image classification` (generic, no domain or competition signal)
**Good query:** `Kaggle medical CT classification 1st place writeup 2024 2025` or `HuggingFace cookbook NER legal contracts 2025`

### Step 2 — Search

Use `WebSearch`. Budget: **at most 3 queries per invocation.**

### Step 3 — Read

For the most promising 3-5 results, use `WebFetch`. Budget: **at most 5 fetches per invocation.**

Prioritize in this order:
1. Kaggle competition discussion threads tagged `1st place`, `2nd place`, `3rd place` (winner solutions).
2. HuggingFace cookbook posts (`huggingface.co/learn/cookbook/...`).
3. HuggingFace blog posts about specific recipes (`huggingface.co/blog/...`).
4. NVIDIA Kaggle Grandmasters Playbook posts.
5. Personal blogs of known competition grandmasters (Abhishek Thakur, Philipp Schmid, Chris Deotte, Psi, Bestfitting, etc.).

Skip:
- Tutorials that don't reference real competition results.
- Marketing posts.
- Q&A sites with no upvoted answers.
- Anything older than 3 years for fast-moving topics (LLM, VLM, SAM family).

### Step 4 — Synthesize the playbook

Read everything. Extract three categories of patterns:

1. **What winners consistently did** — the things 3+ writeups all mention (e.g., "all top-5 solutions used 5-fold stratified CV", "everyone used mixup + cutmix").
2. **Where winners diverged** — the legitimate forks and the tradeoffs they reported (e.g., "1st place used distillation; 2nd place ensembled raw without distillation; both worked, distillation was 5x cheaper to serve").
3. **What winners tried and dropped** — negative results (e.g., "two solutions tried test-time MOE; both dropped it for inference latency"). These are gold and rarely appear in papers.

Do NOT return a citation table. Internalize the evidence.

### Step 5 — Return the playbook

Output exactly this format:

```
## Similar problems found

- <one-line summary of solution 1, with year>
- <one-line summary of solution 2, with year>
- <one-line summary of solution 3, with year>
[2-5 entries — name them by competition or cookbook title in plain language, NEVER fabricate URLs]

## What winners consistently do

- <pattern 1 — appearing in 3+ writeups>
- <pattern 2>
- <pattern 3>
[3-5 bullets — ordered by how reliably they appeared]

## Where winners disagree

- **<fork name>:** <option A> vs <option B>. Tradeoff: <one-sentence why each is chosen>.
- **<fork name>:** <option A> vs <option B>. Tradeoff: <...>.
[1-3 forks — only include legitimate forks, not random differences]

## What winners tried and dropped

- <negative result 1 with brief why>
- <negative result 2 with brief why>
[0-3 bullets — omit section entirely if nothing surfaces]

## Recommended starting playbook for your problem

1. <step 1 — concrete action grounded in the patterns above>
2. <step 2>
3. <step 3>
4. <step 4>
[4-6 ordered steps tailored to the user's specific data/task, NOT generic "load → train → eval"]

## Confidence

<low | medium | high> — <one-sentence justification grounded in how many similar solutions were found and how strongly they agreed>
```

If `Confidence` is `low`, the orchestrator should invoke `ml-engineer-research` for the open questions or surface alternatives prominently. If `high`, the orchestrator can adopt the recommended playbook directly.

## Hard constraints

- NEVER fabricate competition names, solution authors, dataset names, or URLs. If you did not successfully fetch a source, do not refer to its contents.
- NEVER quote a writeup verbatim. Re-express in your own words.
- NEVER produce a citation list. The user wants the playbook, not the homework.
- NEVER recommend a strategy that no surfaced winner actually used. If your reading is sparse, lower the Confidence and say what's missing.
- NEVER skip the "what winners disagree" section when forks genuinely exist — hiding the disagreement creates false confidence.
- Stay within budget: 3 queries, 5 fetches per invocation. If the question is too broad, narrow it (by data domain, task type, or recency) and re-invoke rather than expanding the budget.
- IF web search is unavailable in this session, return `Confidence: low` with playbook drawn from training-data priors, clearly labeled as such.
- Match the user's domain vocabulary (medical → use clinical terms; finance → returns/Sharpe; etc.). Do NOT force generic ML jargon.

## Recipe template

This skill is invoked from inside Claude (no script generation). The "recipe" is the synthesis pattern above. For consistency, the orchestrator may use the following internal scratchpad while reading:

```
PROBLEM: <user's problem in 1 sentence>

SOLUTIONS FOUND:
- [<source>] <approach summary>: <key choices: backbone, augmentation, loss, ensemble, ...>
- [<source>] <approach summary>: <...>
- [<source>] <approach summary>: <...>

PATTERNS:
- consistent: <list>
- divergent: <list of forks>
- dropped: <list of negative results>

PLAYBOOK FOR USER:
1. ...
```

The scratchpad is internal — do NOT include it in the final output. The final output is the structured playbook from Step 5.

## Research hooks

This skill IS the research hook. It does not delegate to other research skills — it directly invokes `WebSearch` and `WebFetch`. However, the orchestrator should:

- Invoke `ml-engineer-research` BEFORE `dl-prior-art` if the user's problem domain is genuinely unfamiliar (e.g., "what does Kaggle even mean for survival analysis?"). `ml-engineer-research` provides framing; `dl-prior-art` provides playbook.
- Invoke `dl-prior-art` BEFORE picking a backbone family in CV/NLP/LLM sub-agents. Knowing what winners actually used prevents picking out-of-fashion architectures.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- The output has the 5 mandatory sections in order (Similar problems found / What winners consistently do / Where winners disagree / Recommended starting playbook / Confidence). The 6th section "What winners tried and dropped" is optional — present when negative results surfaced, omitted entirely when none did. When omitted, the output flows from "Where winners disagree" directly to "Recommended starting playbook".
- The "Similar problems found" section names at least 2 distinct solutions.
- The "Recommended starting playbook" has 4-6 ordered steps. Each step must contain at least one specific noun (backbone name, augmentation name, loss function name, optimizer name, or numeric hyperparameter range) that ALSO appears in "What winners consistently do" or "Where winners disagree". A playbook step that reads as generic boilerplate ("load data", "train model", "evaluate") fails this gate.
- No URLs, no author names, no quoted text appear in the output.
- `Confidence` is one of `low`, `medium`, `high` and has a one-sentence justification.

## Output checklist

- [ ] Used `WebSearch` (≤3 queries) and `WebFetch` (≤5 reads)
- [ ] Output has all 5 (or 6) prescribed sections in order
- [ ] At least 2 distinct similar solutions named
- [ ] "What winners consistently do" has 3-5 patterns appearing in multiple writeups
- [ ] "Where winners disagree" included when forks exist
- [ ] "What winners tried and dropped" included if any negative results surfaced
- [ ] Recommended playbook is tailored to the user's specific data/task
- [ ] Confidence stated with one-sentence justification
- [ ] No URLs, paper titles, author names, or quoted text in the output
- [ ] Vocabulary matches the user's domain
