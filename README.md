# ML Engineer (Claude Code Plugin)

A local-only data-science / ML / quant assistant for Claude Code. Plans, researches, writes, executes, verifies, debugs, and reviews Python work in an isolated venv on your machine. Domain-agnostic — works for ML, finance, healthcare, drug discovery, retail, forecasting, ops research, and any quantitative discipline. No cloud services beyond Claude Code itself.

## What's in here

```
ML_Engineer/
├── .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json
├── agents/
│   ├── ml-engineer.md              # router + tabular orchestrator
│   ├── cv-engineer.md              # vision sub-orchestrator
│   ├── nlp-engineer.md             # NLP encoder sub-orchestrator
│   └── llm-engineer.md             # LLM/VLM sub-orchestrator
├── skills/
│   ├── ml-engineer-*               # 15 tabular/quant skills (unchanged from v0.1.0)
│   └── dl-*                        # NEW deep-learning skills (Phase 1+2 ships 21, total target 33)
│       ├── dl-detect-env/          # probes local compute + remote handoffs, writes env.json
│       ├── dl-remote-execute/      # 7-provider dispatcher (Modal/RunPod/Vast/Lambda/Beam/SSH/Colab)
│       ├── dl-experiment-track/    # wandb / mlflow / aim wiring
│       ├── dl-checkpoint/          # save / resume / FSDP2-aware sharding
│       ├── dl-distributed/         # single-GPU vs FSDP2 vs DeepSpeed selector
│       ├── dl-debug-training/      # NaN / OOM / divergence root-cause triage
│       ├── dl-prior-art/           # Kaggle winner / HF cookbook lookup for similar problems
│       ├── dl-load-data/           # HF datasets / image folder / webdataset / jsonl loader, NLP tokenizer policy
│       ├── dl-augment/             # CV (albumentations + mixup) / NLP (back-translation, MLM, EDA) per use case
│       ├── dl-finetune-loop/       # Trainer-vs-Accelerate selector with smoke-test gate
│       ├── dl-cv-classify/         # timm backbone finetune
│       ├── dl-cv-detect/           # detection finetune (no baked backbone; runtime user choice)
│       ├── dl-cv-segment/          # SAM/YOLO-seg/U-Net family with mask-aware augmentation
│       ├── dl-cv-eval-classify/    # top-k accuracy / per-class F1 / confusion matrix / ECE
│       ├── dl-cv-eval-detect/      # COCO-style mAP via pycocotools
│       ├── dl-cv-eval-segment/     # mIoU / Dice / Hausdorff / boundary F1
│       ├── dl-nlp-classify/        # encoder finetune (ModernBERT/DeBERTa-v3 default)
│       ├── dl-nlp-token/           # NER / token classification with BIO alignment verification
│       ├── dl-nlp-eval-classify/   # accuracy / F1 modes / MCC / ECE
│       ├── dl-nlp-eval-token/      # span-F1 (seqeval) / error analysis (FP/FN/boundary)
│       └── dl-nlp-eval-generative/ # ROUGE / BLEU / BERTScore / perplexity
├── docs/superpowers/
│   ├── specs/                      # design specs (committed before implementation)
│   └── plans/                      # implementation plans (per-phase)
└── README.md
```

## Install (local development)

```bash
# From any Claude Code session in this directory:
/plugin marketplace add /Users/lokesh/Desktop/code/Bio_hacking/ML_Engineer
/plugin install ml-engineer
```

Or directly:

```bash
claude --plugin-dir /Users/lokesh/Desktop/code/Bio_hacking/ML_Engineer
```

## How it works

The `ml-engineer` subagent activates whenever you describe a quantitative task and drives a deterministic loop:

```
research → decide → plan → write-code → execute → verify → debug (if needed) → review
```

It runs **autonomously**. You can interrupt at any time, but the agent does not pause for plan approval — absence of interruption is consent. The first time the venv is needed it asks once, then is silent on subsequent runs.

Per-step:

1. Writes a script to `./newton_workdir/<UTC-timestamp>/step_N_<name>.py`.
2. Runs it under the venv at `${CLAUDE_PLUGIN_DATA}/venv` (falls back to `~/.claude/ml-engineer/venv`).
3. On failure, `ml-engineer-debug` performs root-cause analysis (not symptom patching) and returns a corrected script. After 3 failed fixes on the same step, it stops patching and audits the data pipeline / assumptions / architecture instead.
4. On success, `ml-engineer-verify` runs a separate check via a different code path — exit-0 is not verification.

Before declaring the whole task complete, `ml-engineer-review` runs a fresh-eyes critique covering plan-vs-result drift, methodological soundness (walk-forward CV, transaction costs, multiple-testing correction, scaffold splits, assumption checks), reproducibility, and honesty of the result.

## Deep learning support (v0.2.0-alpha.2, Phase 1 + Phase 2)

Beyond tabular ML, the `ml-engineer` agent now routes deep-learning tasks (CV, NLP, LLM, VLM) to dedicated sub-agents:

- `cv-engineer` — image classification, detection, segmentation
- `nlp-engineer` — encoder fine-tuning (classification, NER, QA, embeddings)
- `llm-engineer` — LLM and VLM finetuning, preference tuning, eval, merge, quantize, serve

Each sub-agent runs the same disciplined loop as the tabular orchestrator (research → decide → plan → write → execute → verify → debug → review) with domain-appropriate skills.

**Phase 1 status (this release):** Seven shared infrastructure skills shipped:

- `dl-detect-env` — probe compute fleet (CPU/MPS/CUDA + Modal/RunPod/Vast/Lambda/Beam/SSH/Colab handoffs) and write `<workdir>/env.json`.
- `dl-remote-execute` — dispatcher across 7 providers with "ask once, continue silently" model and per-step cost surfacing.
- `dl-experiment-track` — wire wandb / mlflow / aim before training starts (with auth verification and HF Trainer integration), so no run is untracked.
- `dl-checkpoint` — save / resume with FSDP2-aware sharding, PEFT adapter-only saves, ephemeral-remote persistence hooks.
- `dl-distributed` — pick single-GPU vs FSDP2 vs DeepSpeed ZeRO-3 by VRAM budget.
- `dl-debug-training` — 5-phase root-cause triage for NaN / OOM / divergence / degenerate output (read failure → form hypotheses → probe → smallest fix → 3-failure escape hatch / pipeline audit).
- `dl-prior-art` — search the web for Kaggle competition winners and HuggingFace cookbook posts on similar problems; return a structured "what did winners do" playbook.

**Phase 2 status (this release):** End-to-end CV and NLP training + evaluation now work. Fourteen new skills:

- `dl-load-data` — HF datasets, image folders, webdataset, jsonl/csv; locks tokenizer + max_length policy for NLP, image_size for CV.
- `dl-augment` — CV (albumentations + mixup/cutmix) and NLP (back-translation, MLM noise, EDA) per use case; bbox-aware for detection, mask-aware for segmentation.
- `dl-finetune-loop` — Trainer-vs-Accelerate selector based on task complexity; mixed precision per detected device; 10-step smoke-test gate.
- `dl-cv-classify` / `dl-cv-detect` / `dl-cv-segment` — timm finetune / runtime-backbone-pick detection / SAM-YOLO-seg-Unet segmentation.
- `dl-cv-eval-classify` / `dl-cv-eval-detect` / `dl-cv-eval-segment` — split per task: top-k+ECE / COCO-mAP / mIoU+Dice+HD95.
- `dl-nlp-classify` / `dl-nlp-token` — encoder classification / NER with mandatory BIO alignment verification.
- `dl-nlp-eval-classify` / `dl-nlp-eval-token` / `dl-nlp-eval-generative` — split per task: F1+MCC+ECE / seqeval span-F1 with error analysis / ROUGE+BLEU+BERTScore.

LLM/VLM-specific skills (Unsloth/Axolotl LoRA, DPO/GRPO, mergekit, AWQ, VLM finetune) ship in Phase 3. Until then, the `llm-engineer` sub-agent uses the generic `dl-finetune-loop` Trainer path as fallback.

**Remote execution.** `dl-detect-env` probes for configured remote providers; `dl-remote-execute` shows the user the top-3 candidates with cost + latency tradeoffs and runs the script on the chosen provider, fetching results back to the local workdir. The user picks once at the start of a remote chain; subsequent remote steps continue silently on the same provider until the user switches or the resource requirement changes.

**Prior-art lookup.** `dl-prior-art` is a Tier 1 shared infrastructure skill. Useful for any new problem (tabular or DL) — invoke it early to surface what Kaggle winners and HF cookbook authors actually did on similar problems. Returns a structured playbook (similar problems found / what winners consistently do / where winners disagree / what winners tried and dropped / recommended starting playbook).

See `docs/superpowers/specs/2026-05-01-dl-skills-design.md` for the full design and `docs/superpowers/plans/2026-05-02-dl-skills-phase-1-foundation.md` for the Phase 1 implementation plan.

## Venv

- Location: `${CLAUDE_PLUGIN_DATA}/venv`, falls back to `~/.claude/ml-engineer/venv`.
- Created on first use with one-time user approval.
- Empty by default — packages installed on demand when scripts raise `ModuleNotFoundError` (per-install approval).
- Shared across all sessions.

## Workdir

Each task writes to `./newton_workdir/<UTC-timestamp>/` in the user's current directory:

```
newton_workdir/2026-04-29T14-22-09Z/
├── step_1_load.py
├── verify_step_1.py
├── step_2_explore.py
├── verify_step_2.py
├── output_summary.txt
└── charts/
    ├── correlation_matrix.png
    └── target_distribution.png
```

The workdir belongs to the user; the plugin never deletes it.

## Hard constraints (enforced by the skills)

- No `plt.show()`, `input()`, web servers, infinite loops, or destructive file ops in generated code.
- No system Python, no global pip, no `sudo`.
- Code only runs from inside `newton_workdir/<timestamp>/`.
- Charts saved to `charts/` and announced via `print("Chart saved as ...")`.
- No completion claim without fresh verification evidence.
- No fix without root-cause investigation first.

## Inspiration & credit

The skill structure, "Iron Law" framing, trigger-only descriptions, severity-tagged review pattern, and 4-phase debug methodology were inspired by [obra/superpowers](https://github.com/obra/superpowers) — Jesse Vincent's agentic skills framework. We borrowed the patterns most relevant to a data-science workflow (verification discipline, root-cause debugging, end-of-task review, autonomous orchestration via skill descriptions) and skipped the parts that didn't fit (TDD-as-religion, git-worktree ceremony, branch-finishing flow).

The CV-first / metric-first / project-layout / leakage-pattern discipline in `ml-engineer-cv-design`, `ml-engineer-pick-metric`, the Layout B project skeleton in `ml-engineer-write-code`, and the ML-specific failure patterns added to `ml-engineer-verify` follow the methodology from **Abhishek Thakur's *Approaching (Almost) Any Machine Learning Problem*** — specifically the cross-validation, evaluation metrics, arranging-ML-projects, categorical variables, feature selection, hyperparameter optimization, and ensembling chapters. The recipes (Sturge's rule for binning regression targets, GroupKFold for grouped data, AUC≈1 → suspect, target-encoding-only-inside-folds, the `train.py --fold N --model X` idiom, the per-model HPO ranges) are his.

The plugin shape itself follows Anthropic's official [Claude Code plugin specification](https://code.claude.com/docs/en/plugins-reference) and [Skills authoring guide](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf).

## License

MIT
