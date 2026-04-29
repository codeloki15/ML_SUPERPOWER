---
name: ml-engineer-hypothesis
description: Generates testable hypotheses for any quantitative problem with mechanism, prediction, experiment design, and falsification criteria. Use across domains — ML, finance, biology, social science, ops, healthcare — when the user asks to "propose hypotheses", "design experiments", "what could explain this result", "what could be tested next", or after research has been done and the user wants to extend findings. Refuses to fabricate evidence.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Hypothesis Generator

Produce hypotheses that are **testable in this session** with the user's data and the existing plan→write→execute loop. Bad hypotheses are vague, untestable, or require infrastructure the user doesn't have. Good hypotheses can be falsified by a script that runs in under a few minutes.

Domain-agnostic. Works for an ML engineer asking "why is recall low", a quant asking "why does this strategy underperform OOS", an epidemiologist asking "why are these subgroups different", a retail analyst asking "why is forecast error spiking on Mondays".

## When to invoke

- "What could explain why <metric / result / observation> is bad / surprising?"
- "What should we test next?"
- "Generate hypotheses for <observation>."
- After `ml-engineer-research` returned a `low` confidence — multiple hypotheses worth testing rather than one decision to make.

## Process

### Step 1 — Anchor

Identify what's known:
- The user's actual data and task
- Any results / metrics already produced this session
- Any conclusion from `ml-engineer-research` (in context, no need to re-run)

### Step 2 — Generate

Produce **3-5 hypotheses**, ranked by `risk` × `informativeness`. The best hypothesis is one that's cheap to test and would meaningfully update beliefs either way.

### Step 3 — Output

For each hypothesis, use exactly this format:

```
### H<N>: <one-sentence hypothesis, falsifiable>

- **Mechanism:** <2 sentences — why we'd expect this to be true>
- **Prediction:** <a specific, measurable outcome — quantitative wherever possible>
- **Experiment:** <minimal script that tests it — what data, what method, what metric>
- **Falsification:** <what observation would refute this hypothesis>
- **Cost:** <rough — seconds, minutes, hours of compute>
- **Information value:** <high | medium | low — does the result meaningfully change what we'd do next>
```

Then end with:

```
## Recommended next test

H<N> — <one sentence why this is the best one to run first>
```

## Domain-neutral examples of good predictions

A prediction must specify *what changes by how much*, not just "improves":

- ML: `validation F1 increases by ≥0.05`
- Finance: `out-of-sample Sharpe improves from 0.6 to >1.0 with the proposed feature dropped`
- Healthcare: `subgroup A's hazard ratio drops below 1.2 after adjusting for confounder X`
- Forecasting: `weekly sMAPE on Mondays drops below the all-week average`
- Retail: `forecast error variance halves when promo flag is included`

The pattern is the same across domains: a metric, a direction, a magnitude, a condition.

## Hard rules

- Every hypothesis must be **falsifiable** — there must be a concrete observation that would prove it wrong. Hypotheses like "the result could be improved" are not falsifiable.
- Every hypothesis must be **testable in this session** — using the user's actual data and the existing skills. No "we'd need a different dataset" hypotheses.
- Predictions must be **quantitative** wherever possible. A specific number with units beats an adjective.
- **Do not fabricate evidence.** Mechanisms can reference general domain principles; they cannot cite specific paper titles, author names, or URLs that you did not actually fetch in this session.
- If the user has no data or no prior result, ask for context before generating. Don't generate hypotheses in a vacuum.
- Use the user's domain vocabulary. If they said "alpha", don't translate it to "predictive signal"; if they said "C-index", don't switch to "ranking accuracy".

## Anti-patterns to avoid

- **Generic wisdom:** "More data would help." (Not a testable hypothesis.)
- **Untestable:** "The model is overfitting" / "The strategy is curve-fit." Make it concrete: "If <train metric> - <val metric> > <threshold>, then overfit; reducing <hyperparameter> from X→Y should narrow the gap."
- **Bundled:** "Try A, B, and C." That's three hypotheses. Split them.
- **Citation theatre:** "Smith et al. 2024 found that..." Don't.

## Output checklist

- [ ] 3-5 hypotheses, each with all six fields
- [ ] Each hypothesis is one sentence and falsifiable
- [ ] Predictions are quantitative
- [ ] No fabricated paper titles or author names
- [ ] One recommended next test named at the end
- [ ] All hypotheses testable with the user's existing data this session
- [ ] Vocabulary matches the user's domain
