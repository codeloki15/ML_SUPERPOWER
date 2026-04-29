---
name: ml-engineer-plan
description: Use when the user asks to "analyze", "explore", "model", "forecast", "train", "backtest", "simulate", "evaluate", "predict", "visualize", "summarize", or "investigate" data — especially with a .csv, .xlsx, .json, or .parquet file. Do NOT use for one-line code edits, general programming questions, or non-data tasks.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# ML Engineer — Plan

Produce a hierarchical, checkbox TODO plan in GitHub-flavored markdown. No code. No prose paragraphs.

## Required structure

```
## Goal
One sentence stating what the user wants.

## Plan
### 1. Data loading
- [ ] Load <file path> into a DataFrame
- [ ] Print shape, dtypes, head(5)

### 2. Exploration
- [ ] ...

### 3. <next phase>
- [ ] ...

## Assumptions
- Bullet 1
- Bullet 2
```

## Rules

- Use `- [ ]` for every actionable step. No bullets without checkboxes inside `## Plan`.
- Section headers (`### 1.`, `### 2.`) group related steps. Number them.
- Each step is one short imperative sentence. No "we will…", no rationale inside the step.
- Put trade-offs and assumptions in the `## Assumptions` block, not inside steps.
- Never include code blocks. The plan is intent, not implementation.
- If the task needs a data file and the user did not name one, ask exactly once: `Which file should I use? (path)` and stop until they answer.
- If the user is updating a prior plan, prefix changed steps with `**[UPDATED]**`, new steps with `**[NEW]**`, and removed steps with `**[REMOVED]**`. Open with a `### Plan Updates:` summary listing what changed.

## Phasing guidance

For most quantitative tasks, the phases are some subset of:

1. Data loading & sanity checks
2. Exploration (shape, dtypes, missing values, distributions, time coverage)
3. Cleaning / transformation
4. Feature / variable construction (if modeling)
5. Method application (modeling, statistical test, simulation, backtest, forecast — whatever the task requires)
6. Evaluation (with the metric appropriate to the domain — accuracy/F1, Sharpe/drawdown, C-index, sMAPE, lift, p-value, etc.)
7. Reporting / visualization

Skip phases that don't apply. Don't invent phases the task doesn't need. Match phase 5/6 vocabulary to the user's domain — "train a model" for ML, "fit a hazard model" for survival, "build a backtest" for finance, "fit a forecast" for retail, etc.

## Output checklist

Before returning, verify:

- [ ] Every step under `## Plan` uses `- [ ]`
- [ ] Section headers are numbered
- [ ] No code blocks anywhere
- [ ] File path stated explicitly in step 1 (or the "ask once" question was used)
- [ ] `## Assumptions` block present with at least one bullet
