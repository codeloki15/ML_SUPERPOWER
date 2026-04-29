---
name: ml-engineer-research
description: Use when facing an unfamiliar problem class, choosing between methods or metrics, picking starting parameters, diagnosing implausible results, or when the user explicitly asks to "look up", "research", "check the literature", or "find recent work on" a topic. Do NOT use for basic questions with known answers or topics already covered in the current session.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Research

Search the web, read what's relevant, form an informed opinion. Do not paste citation tables back to the user — internalize the evidence and pass a conclusion to `ml-engineer-decide`.

This skill is domain-agnostic. It works for ML choices, finance methodology, biostatistics, drug-screening pipelines, demand-forecasting techniques, ops research — anywhere a quantitative decision benefits from current best practice.

## When to invoke

- Choosing a method, metric, evaluation scheme, sampling strategy, or starting parameters for a problem you don't have a strong prior on.
- Diagnosing why a result is implausible, unstable, or fails a sanity check.
- The user explicitly asks to "look up", "research", "check the literature", "find recent work on...".

## When NOT to invoke

- The user is asking a basic question with a known answer.
- A previous research pass in the same session already covered the topic — reuse what's in context.
- The decision is trivial (chart color, file name).

## Process

### Step 1 — Frame the question

Rewrite the open problem as 2-3 specific search queries. Include:

- The technique class (whatever the user is working in — modeling, evaluation, signal processing, simulation, etc.)
- The data shape / domain qualifier the user mentioned (e.g. "small sample", "high-frequency", "longitudinal", "censored", "panel data", "hierarchical")
- A recency filter: prefer results from the last 3 years. Append `2024 OR 2025 OR 2026` if the topic is moving fast. Skip the filter for foundational, stable topics.

Bad query: `<topic>` (one word, no qualifiers)
Good query: `<technique> <data shape> <domain qualifier> 2024 best practices`

The query templates are intentionally domain-neutral. Substitute the user's actual vocabulary.

### Step 2 — Search

Use the `WebSearch` tool. Budget: **at most 3 queries per invocation.**

### Step 3 — Read

For the most promising 3-5 results across all queries, use `WebFetch` to read the actual content. Budget: **at most 5 fetches per invocation.**

Skip results that are:

- Older than 3 years unless the topic is foundational and stable
- Q&A sites with no answers or zero-vote answers
- Marketing pages, vendor blogs without technical content
- Behind paywalls (the fetch will fail or return a stub)

### Step 4 — Synthesize silently

Read everything. Form a private mental model. Note conflicts, recency, and applicability to the user's specific situation.

**Do not return a citation table to the user.** Do not say "Source A says X, Source B says Y." The user wants the conclusion, not the homework.

### Step 5 — Return a conclusion

Output exactly this format:

```
## Research summary

<1-2 sentences: what the consensus or leading approach is for this specific situation>

## Conclusion

<the recommended approach, in plain language, applied to the user's actual problem — 2-4 sentences>

## Caveats

- <one thing that could change the recommendation>
- <one thing that conflicts in the literature, if any>
- <one thing about the user's data we'd need to verify before committing>

## Confidence

<low | medium | high> — <one-sentence justification>
```

If `Confidence` is `low`, the next skill (`ml-engineer-decide`) should surface alternatives prominently. If `high`, it can act on the recommendation directly.

## Hard rules

- Never fabricate paper titles, author names, or URLs. If you didn't successfully fetch a page, do not refer to its contents.
- Never quote a source. Re-express in your own words.
- Never produce a citation list. The user does not want one.
- Stay within the budget: 3 queries, 5 fetches. If the question is too broad, narrow it and re-invoke rather than expanding the budget.
- If web search is unavailable in this session, say so and return `Confidence: low` with reasoning from training-data priors clearly labeled as such.
- Match the user's domain vocabulary. If they're in finance, talk about returns and Sharpe; if survival analysis, talk about hazard ratios and C-index; if forecasting, talk about sMAPE and seasonality. Don't force ML jargon onto non-ML problems.

## Output checklist

- [ ] Used `WebSearch` (≤3 queries) and `WebFetch` (≤5 reads)
- [ ] Output has `## Research summary`, `## Conclusion`, `## Caveats`, `## Confidence`
- [ ] No URLs, no paper titles, no author names in the output
- [ ] Conclusion applies to the user's specific situation, not a generic answer
- [ ] Vocabulary matches the user's domain
- [ ] Caveats name at least one thing that would change the recommendation
