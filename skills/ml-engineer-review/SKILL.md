---
name: ml-engineer-review
description: Use before declaring a multi-step task complete, before merging a result into the user's "final" answer, or when the orchestrator wants a fresh-eyes critique of a plan, script, or result. Do NOT use as a substitute for ml-engineer-verify (verify is per-step evidence; review is end-of-task critique).
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Review

End-of-task critique. Run before claiming a multi-step task complete. Catches the things `ml-engineer-verify` doesn't — design-level mistakes, plan-vs-result drift, methodology violations, missing follow-ups.

`ml-engineer-verify` answers *"did this step do what it claimed?"* — local, mechanical, per-step.
`ml-engineer-review` answers *"is the whole result actually defensible?"* — global, methodological, end-of-task.

Both are required before completion.

## When to invoke

- Just before reporting a multi-step task complete to the user.
- After a major step that locks in a methodological choice (model selected, evaluation scheme set, backtest finalized).
- When the user asks "is this right" or "review this" or "look this over".

## What to review (severity-tagged)

For each item below, classify findings into one of three severities:

- **Critical** — methodologically wrong, results not defensible, must fix before completion.
- **Important** — likely wrong or risky, fix recommended; if user accepts the risk, document it.
- **Minor** — cosmetic, stylistic, or nice-to-have.

### Plan-vs-result alignment

- Did every plan step actually run, or were some silently skipped?
- Do final outputs match what the plan promised (file names, metrics named, charts produced)?
- Were any phases inflated or truncated relative to the plan?

### Methodological soundness

For modeling tasks:
- Train / validation / test split scheme appropriate for the data structure (random vs stratified vs grouped vs walk-forward vs scaffold)?
- Preprocessing fit only on training data?
- Metric appropriate for the task and class balance?
- Confidence intervals or repeat runs where the result hinges on a single number?

For backtests:
- Walk-forward, not k-fold?
- Transaction costs and slippage included?
- Survivorship bias considered?
- Lookahead audit run on every feature?

For statistical tests:
- Sample size adequate for the test?
- Multiple-testing correction applied if warranted?
- Assumptions (normality, independence, equal variance) checked?
- Effect size reported alongside p-value?

For forecasts:
- Horizon and frequency stated and consistent?
- Held-out period genuinely held out?
- Naive baseline (seasonal-naive, last-value) compared against?

### Reproducibility

- Random seed fixed for any stochastic step?
- Data version / source / timestamp recorded?
- Library versions captured?
- Workdir contents sufficient to re-run from scratch?

### Honesty of the result

- Headline metric reflects the worst-case interpretation, not the best-case?
- Per-class / per-subgroup performance reported when relevant?
- Failure modes characterized (where does the model break, where do hypotheses fail to hold)?
- Caveats and limitations stated, not buried?

### Code hygiene (lower priority)

- Any `try/except` swallowing errors silently?
- Magic numbers without rationale?
- Dead code from earlier iterations?

## Process

### Step 1 — Read the workdir

List `<workdir>/`. Read every `step_<N>_*.py` and `verify_step_<N>.py`. Read the plan output. Read the final summary.

### Step 2 — Cross-check

For each section above, find evidence in the workdir for or against it. Be specific — cite filenames and line numbers.

### Step 3 — Classify

Each finding gets a severity. **At minimum, scan all of "Methodological soundness" and "Honesty of the result."** Skip a section only if it doesn't apply (no need to check backtests for a one-off statistical test).

### Step 4 — Output

Reply in exactly this format:

```
## Review of <task name>

### Critical
- <finding 1, with file:line reference and one-line fix>
- <finding 2>
(Or: "None.")

### Important
- <finding 1, with file:line and rationale>
(Or: "None.")

### Minor
- <finding 1>
(Or: "None.")

### What I checked
- Plan-vs-result alignment
- Methodological soundness (<which subset applied>)
- Reproducibility
- Honesty of the result
- (skip sections that didn't apply)

### Verdict
<release | release-with-caveats | block>
```

### Step 5 — Branch

- **release** → orchestrator proceeds to declare task complete.
- **release-with-caveats** → orchestrator includes the Important findings in the user-facing summary as known limitations.
- **block** → at least one Critical finding. Hand back to debug or plan; do not declare complete.

## Hard rules

- **Be specific.** "Methodology looks off" is not a finding. "Walk-forward CV not used in step_4_backtest.py:33; k-fold leaks future returns into training" is a finding.
- **Cite files and line numbers** for every Critical or Important finding.
- **Don't invent problems.** If a section is fine, say "None" — padding the review with nitpicks dilutes the real findings.
- **Don't repeat what verify already found.** If a step's verification was `failed` and got patched, don't re-flag it.
- **Default to skepticism on suspiciously good results.** A 0.99 R², a 3.5 Sharpe, or a p < 0.001 deserves a Critical-severity audit by default; downgrade to Important or Minor only if the audit clears it.

## Anti-patterns to avoid

- **Performative review.** Listing 12 minor style points to look thorough. Real reviews are short and pointed.
- **Avoiding the hard call.** If methodology is wrong, say `block`. Don't soften to `release-with-caveats` to avoid friction.
- **Reviewing the orchestrator's reasoning instead of the artifacts.** The artifacts are the ground truth. The orchestrator's narrative is not.

## Output checklist

- [ ] Read the actual files in the workdir (not the orchestrator's summary)
- [ ] Critical / Important / Minor sections each present (with "None" if empty)
- [ ] Every Critical / Important finding cites a file and line
- [ ] Verdict is one of: release | release-with-caveats | block
- [ ] If suspiciously-good result, defaulted to skeptical audit
