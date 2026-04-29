---
name: ml-engineer-decide
description: Use after ml-engineer-research returns a conclusion, when the user shares results and asks "what next", or when the orchestrator hits a methodological fork (model class, loss/objective, sampling scheme, primary metric, split scheme, backtest design). Do NOT use for routine choices (chart color, file name) or decisions already locked by an approved plan.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# ML Engineer — Decide

Convert evidence into a concrete next action. Make the reasoning auditable so the user can push back.

## When to invoke

- Right after `ml-engineer-research` returns a conclusion.
- When the user shares results / metrics and asks "what next".
- When mid-loop the orchestrator hits an architectural fork (model family, loss function, eval metric, sampling strategy, train/val split scheme).

## When NOT to invoke

- For routine choices (chart color, variable name, file name). Just pick.
- When the answer is already in the approved plan. Just execute.

## Process

### Step 1 — Restate the question

In one sentence, what decision is being made and what's at stake.

### Step 2 — Decide

Pick the recommended option. Don't equivocate.

### Step 3 — Output

Use exactly this format:

```
## Decision: <the chosen option, one line>

## Why this, applied to your situation
<2-4 sentences. Reference concrete properties of the user's task: dataset size,
class balance, compute budget, prior result, etc. Generic justifications are
not allowed.>

## Alternatives I considered

1. **<alt 1>** — <one sentence why I rejected it for this case>
2. **<alt 2>** — <one sentence why I rejected it for this case>

## Risks of going with the recommendation

- <concrete risk #1, with a one-line mitigation>
- <concrete risk #2, with a one-line mitigation>

## What would change my mind

- <a specific finding from running this approach that would push me to alt 1>
- <a specific data property we haven't yet confirmed that would push me to alt 2>

## Verification before we trust this

<one or two checks the orchestrator should run after the action to confirm
it actually worked — see ml-engineer-verify>
```

### Step 4 — Approval gate

Categorize the decision:

| Category | Examples | Approval |
|---|---|---|
| **Architectural / methodological** | Method family (model class, statistical test, simulation type), loss / objective / fit criterion, sampling or splitting scheme, primary evaluation metric, backtest or experiment design | **Required** |
| **Parameter** | learning rate, n_estimators, smoothing factor, lookback window, lag length, regularization strength | Skipped |
| **Routine** | chart style, log verbosity, file names | Skipped |

If **Architectural / methodological**: state `Approval required to proceed` at the bottom and stop. Wait for user.

If **Parameter** or **Routine**: state `Proceeding` at the bottom and let the orchestrator continue.

## Hard rules

- Pick one recommendation. "It depends" is not an answer — if it truly depends, run more research first.
- The "Why this, applied to your situation" section must reference at least one concrete property of the user's actual data or task. Generic ML wisdom doesn't count.
- The "What would change my mind" section must contain at least 2 items, each falsifiable by something the orchestrator can actually observe.
- The "Verification" section must propose checks that don't take longer than the action itself. Cheap, fast verification only.
- Never say "based on research" or "according to the literature" — the conclusion is yours, own it.

## Output checklist

- [ ] One concrete recommendation, not a list of options dressed as a recommendation
- [ ] Justification cites properties of the user's task, not generic wisdom
- [ ] At least 2 alternatives with reasons rejected
- [ ] At least 2 "what would change my mind" items
- [ ] Verification plan is concrete and fast
- [ ] Categorized correctly (architectural → ask, otherwise → proceed)
