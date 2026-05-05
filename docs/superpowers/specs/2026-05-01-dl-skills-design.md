# Deep Learning Skills Extension — Design Spec

**Date:** 2026-05-01
**Plugin:** `ml-engineer`
**Version target:** `0.2.0`
**Status:** Approved design, pending implementation plan

## Problem

The `ml-engineer` plugin currently covers tabular / quantitative ML through a deterministic loop (research → decide → plan → write → execute → verify → debug → review) inspired by Abhishek Thakur's *Approaching (Almost) Any Machine Learning Problem*. It does not cover deep learning workflows: computer vision, NLP, LLM training and finetuning, or vision-language models. Users working on those tasks today get no plugin support.

We need to extend the plugin so the same disciplined loop applies to DL work, while respecting the realities of DL: pretrained backbones, GPU lifecycle, tokenizers, augmentation pipelines, LoRA / QLoRA, evaluation harnesses, and remote compute providers (Modal, RunPod, Vast, Lambda, Beam, SSH, Colab).

## Goal

Ship one unified plugin that handles tabular, vision, NLP, LLM, and VLM tasks. Knowledge sources are HuggingFace docs/blogs/cookbooks and recent (2024–2026) Kaggle winner writeups. The new skills encode *process* (when to act, how to set up the workflow) and delegate *facts* (current best LR, current `target_modules`, current SOTA) to runtime web research via the existing `ml-engineer-research` skill.

## Non-goals (v1)

- Production serving infrastructure (vLLM in prod, autoscaling, monitoring). Inference serving is scoped to eval-time only.
- RAG pipelines.
- Custom training loops written from scratch. Skills assume HF Trainer / Accelerate / Unsloth / Axolotl.
- Audio, graph, time-series-DL, recommender-DL. Only the four covered domains: tabular, CV, NLP, LLM/VLM.
- Multi-node distributed training. Single-node FSDP2 / DeepSpeed ZeRO-3 only.
- Encoded version pins for ML libraries — handled by the user's venv.

## Architecture

One unified plugin (`ml-engineer`). One router agent + three domain sub-agents.

```
ML_Engineer/
├── .claude-plugin/
│   ├── plugin.json                 (bumped to 0.2.0)
│   └── marketplace.json
├── agents/
│   ├── ml-engineer.md              (router + tabular loop — existing, modified)
│   ├── cv-engineer.md              (NEW — vision sub-loop)
│   ├── nlp-engineer.md             (NEW — NLP sub-loop)
│   └── llm-engineer.md             (NEW — LLM/VLM sub-loop)
├── skills/
│   ├── ml-engineer-*               (existing 15 — unchanged)
│   └── dl-*                        (NEW 29 skills, see taxonomy below)
└── README.md                       (updated)
```

### Routing

The existing `ml-engineer.md` agent gets a router prologue that runs before the tabular loop. Routing logic:

1. **Strong signal → direct dispatch.** User uploads `.jpg/.png/.tif` → CV. User uploads `.txt/.csv` of text plus says "classify" → NLP. User says "finetune Llama / Qwen / Mistral / GPT-2 / Gemma" or names any HF model ID → LLM. User uploads `.csv` with numeric/categorical → tabular.
2. **Ambiguous signal → ask one clarifying question.** Multiple choice in one message: "Vision, NLP, LLM, or tabular?"
3. **Multi-domain task → router stays in charge.** E.g., "build a CLIP-style retrieval system." Router picks the dominant domain and delegates to others as needed.

### Sub-agents

Each sub-agent has its own loop, its own skills table (drawn from the shared 28-skill taxonomy), and inherits the same iron rules from the existing `ml-engineer.md`:

- All code runs from `./newton_workdir/<UTC-timestamp>/` under the managed venv.
- Per-step `ml-engineer-verify` after every executed step.
- Three-failure cap on debug retries before stopping for user input.
- Never claim done without `ml-engineer-review` returning `release` or `release-with-caveats`.

### Skill ownership

Skills fall into three ownership tiers:

**Tier 1 — Shared infrastructure (7 skills, every sub-agent uses these):**
- `dl-detect-env`, `dl-remote-execute`, `dl-experiment-track`, `dl-checkpoint`, `dl-distributed`, `dl-debug-training`, `dl-prior-art`.

*Note: `dl-prior-art` was added to Tier 1 mid-Phase-1 in response to user feedback. It is a playbook-lookup primitive (search Kaggle winners + HF cookbook posts on similar problems) — distinct from `ml-engineer-research` which answers method-and-fact questions. The two are complementary; the orchestrator may invoke both for a new problem.*

**Tier 2 — Cross-domain training/data utilities (7 skills, used by 2+ sub-agents):**
- `dl-load-data`, `dl-augment`, `dl-finetune-loop`, `dl-pseudo-label`, `dl-distillation`, `dl-cv-pretrain` (only when CV needs it but the recipes generalize), `dl-ensemble-tta`.

**Tier 3 — Domain-specific (15 skills, owned by one sub-agent):**
- CV-owned: `dl-cv-classify`, `dl-cv-detect`, `dl-cv-segment`, `dl-cv-eval`.
- NLP-owned: `dl-nlp-classify`, `dl-nlp-token`, `dl-nlp-eval`.
- LLM-owned: `dl-llm-lora`, `dl-llm-instruction-tune`, `dl-llm-pref-opt`, `dl-llm-eval`, `dl-llm-merge`, `dl-llm-quantize`, `dl-llm-serve`.
- VLM-owned (under LLM sub-agent): `dl-vlm-finetune`.

Any skill is invokable from any sub-agent or from the tabular orchestrator when relevant. Tier labels describe typical use, not a hard restriction.

## Skill taxonomy (29 skills)

Skills derived from late-2025 / early-2026 research on the HuggingFace ecosystem and Kaggle winner solutions.

### Infrastructure / cross-cutting (7)

| # | Skill | Purpose |
|---|---|---|
| 1 | `dl-detect-env` | Probe CUDA/MPS/CPU, VRAM, installed libs, available remote-handoff CLIs (Modal token, `runpodctl`, `vastai`, SSH configs, Beam, Lambda, Colab). Writes `<workdir>/env.json` listing all reachable environments. Gates everything downstream. |
| 2 | `dl-remote-execute` | Dispatcher across Modal / RunPod / Vast / Lambda / Beam / generic SSH / Colab handoff. Per-environment "ask once, continue silently" model. |
| 3 | `dl-experiment-track` | Wire wandb / mlflow / aim before any non-trivial run. |
| 4 | `dl-checkpoint` | Resume, sharded checkpointing, FSDP2 state-dict gotchas, `save_pretrained` vs raw state dict. |
| 5 | `dl-distributed` | Decision skill: single-GPU (Unsloth) vs FSDP2 (Accelerate) vs DeepSpeed ZeRO-3. Launcher recipes per choice. |
| 6 | `dl-debug-training` | NaN / loss-spike / grad-norm explosion / OOM root-cause triage. |
| 7 | `dl-prior-art` | Web-search Kaggle competition winners + HuggingFace cookbook posts for problems similar to the user's task. Returns a structured playbook (similar problems found / what winners consistently do / where winners disagree / what winners tried and dropped / recommended starting playbook / confidence). Distinct from `ml-engineer-research` (which answers generic method-and-fact questions). |

### Data (3)

| # | Skill | Purpose |
|---|---|---|
| 7 | `dl-load-data` | HF datasets, streaming, webdataset, image folders, text corpora. Tokenization is folded in here as one chapter of data prep. |
| 8 | `dl-augment` | albumentations + mixup / cutmix / mosaic + RandAugment. NLP augmentation (back-translation, MLM noise) included as conditional sub-section. |
| 9 | `dl-pseudo-label` | Confidence-thresholded self-training, k-fold-safe pseudo labels, distill-into-single-model. Recurring Kaggle staple. |

### Training core (3)

| # | Skill | Purpose |
|---|---|---|
| 10 | `dl-finetune-loop` | Generic HF Trainer / Accelerate boilerplate. Mixed precision (bf16/fp16/fp8) folded in as one config section. |
| 11 | `dl-cv-pretrain` | SimCLR / DINO / MAE for self-supervised pretraining. Used only when domain has lots of unlabeled images and pretrained backbones do not transfer (medical, satellite, microscopy). Skill's primary job is recognizing when to skip. |
| 12 | `dl-distillation` | Logit / feature distillation plus reasoning-trace (CoT) distillation. Recurring Kaggle finisher. |

### CV (4)

| # | Skill | Purpose |
|---|---|---|
| 13 | `dl-cv-classify` | timm + linear probe / fine-tune recipes. |
| 14 | `dl-cv-detect` | YOLO11/26 default, RT-DETR alt, Detectron2 for academic baselines. |
| 15 | `dl-cv-segment` | SAM2 / SAM3 zero-shot/promptable, YOLO-seg for real-time, box→mask combos. |
| 16 | `dl-cv-eval` | mAP / IoU / Dice / Hausdorff harness for detection and segmentation; top-k accuracy / confusion matrix for classification. |

### NLP (3)

| # | Skill | Purpose |
|---|---|---|
| 17 | `dl-nlp-classify` | Encoder fine-tune (ModernBERT / DeBERTa-v3 defaults). |
| 18 | `dl-nlp-token` | Token classification / NER / extractive QA. Separate from classify because loss / decoding differs. |
| 19 | `dl-nlp-eval` | F1 / exact-match / ROUGE / BLEU / perplexity harness. |

### LLM (7)

| # | Skill | Purpose |
|---|---|---|
| 20 | `dl-llm-lora` | PEFT / LoRA / QLoRA / DoRA decision tree. **Default: Unsloth single-GPU recipe** (3 Kaggle wins in 2024–2025, 2-5x faster, 80% less memory). Override path: multi-GPU → Axolotl; user explicitly names TRL → TRL. User has flexibility to override at any point. |
| 21 | `dl-llm-instruction-tune` | SFT format conventions, chat templates, packing, response-only masking. Also absorbs encoder-decoder seq2seq use cases (T5/BART) since modern stacks use small instruct LLMs. |
| 22 | `dl-llm-pref-opt` | Method-selection skill: pairwise → DPO, binary thumbs → KTO, single-GPU → ORPO, verifiable reward → GRPO. |
| 23 | `dl-llm-eval` | lm-evaluation-harness + lighteval, with vLLM / SGLang as backends. |
| 24 | `dl-llm-merge` | mergekit (SLERP / TIES / DARE-TIES) decision rules. |
| 25 | `dl-llm-quantize` | Post-training quantization for serving: AWQ default, GPTQ alt, GGUF for llama.cpp. Distinct from QLoRA's training-time bitsandbytes. |
| 26 | `dl-llm-serve` | vLLM (default) / SGLang (RAG/agents) recipes for eval-time and benchmarking inference. Scoped to "serve to benchmark / generate synthetic data," not production serving. |

### VLM (1)

| # | Skill | Purpose |
|---|---|---|
| 27 | `dl-vlm-finetune` | TRL Qwen2-VL cookbook plus Axolotl Qwen2.5-VL / Pixtral / LLaVA / SmolVLM2 recipes. |

### Inference / ensembling (1)

| # | Skill | Purpose |
|---|---|---|
| 28 | `dl-ensemble-tta` | K-fold OOF blending, rank-average, snapshot ensembles, TTA. Cross-domain (CV + NLP + tabular). |

## Loop variants

### CV loop (`cv-engineer.md`)

```
1. Research (conditional)         → ml-engineer-research
2. Decide (conditional)           → ml-engineer-decide
3. Plan                           → ml-engineer-plan
4. Setup workdir + detect env     → dl-detect-env (writes env.json)
5. Lock CV foundations:
   5a. EDA probe (image stats, class balance, resolution histogram)
   5b. CV scheme                  → ml-engineer-cv-design (image-aware)
   5c. Metric                     → ml-engineer-pick-metric
   5d. Augmentation policy        → dl-augment
   5e. Backbone family            → dl-cv-classify | dl-cv-detect | dl-cv-segment
6. Decide compute placement       → dl-detect-env says local? else dl-remote-execute
7. Wire experiment tracking       → dl-experiment-track
8. Train baseline                 → dl-finetune-loop
9. Verify                         → ml-engineer-verify + dl-cv-eval
10. Iterate ladder:
    - Pretrain on unlabeled?      → dl-cv-pretrain (rare)
    - Pseudo-label?               → dl-pseudo-label
    - Distill into smaller model? → dl-distillation
    - Ensemble + TTA              → dl-ensemble-tta
11. Final review                  → ml-engineer-review
```

### NLP loop (`nlp-engineer.md`)

Same shape as CV with these substitutions:
- 5d: tokenizer + max_length policy folded into `dl-load-data`.
- 5e: `dl-nlp-classify` or `dl-nlp-token`.
- 9: `ml-engineer-verify` + `dl-nlp-eval`.
- `dl-augment` step is conditional (back-translation, dropout, MLM-noise — not always applied).

### LLM loop (`llm-engineer.md`)

```
1-4. Same as above
5. Lock LLM foundations:
   5a. EDA probe (token length distribution, format check, dedupe stats)
   5b. Pick base model + size      → dl-detect-env says what fits in VRAM
   5c. Format the data             → dl-load-data (chat templates, packing)
   5d. Pick training method        → dl-llm-lora (default Unsloth single-GPU; user can override)
   5e. Pick eval suite             → dl-llm-eval
6. Decide compute placement       → dl-detect-env + dl-remote-execute + dl-distributed
7. Wire experiment tracking       → dl-experiment-track
8. Train baseline (SFT)           → dl-llm-instruction-tune
9. Verify                         → ml-engineer-verify + dl-llm-eval
10. Iterate ladder:
    - Preference tune             → dl-llm-pref-opt (DPO/KTO/ORPO/GRPO selector)
    - Quantize for serving        → dl-llm-quantize
    - Merge with sibling models   → dl-llm-merge
    - Serve & benchmark           → dl-llm-serve
    - Distill to smaller model    → dl-distillation
11. Checkpoint hygiene throughout → dl-checkpoint
12. Final review                  → ml-engineer-review
```

VLM tasks use the LLM loop with `dl-vlm-finetune` substituted for `dl-llm-instruction-tune` at step 8.

## Environment data model

`dl-detect-env` writes `<workdir>/env.json`. Multiple environments per user are first-class.

```json
{
  "active": "local",
  "environments": {
    "local": {
      "kind": "local",
      "device": "mps",
      "vram_gb": 0,
      "torch_version": "...",
      "available_libs": ["transformers", "peft", "..."]
    },
    "modal": {
      "kind": "remote",
      "provider": "modal",
      "auth": "ok",
      "default_gpu": "A10G",
      "available_gpus": ["T4", "A10G", "A100-40GB", "H100"]
    },
    "runpod-h100": {
      "kind": "remote",
      "provider": "runpod",
      "auth": "ok",
      "ssh_target": "...",
      "vram_gb": 80
    },
    "ssh-gpu-box": {
      "kind": "remote",
      "provider": "ssh",
      "host": "gpu-box.example",
      "device": "cuda",
      "vram_gb": 24
    },
    "vastai": {"kind": "remote", "provider": "vastai", "auth": "ok"},
    "lambda": {"kind": "remote", "provider": "lambda", "auth": "ok"},
    "beam": {"kind": "remote", "provider": "beam", "auth": "ok"},
    "colab": {"kind": "remote", "provider": "colab", "detected_runtime": false}
  }
}
```

`active` tracks which environment the orchestrator is currently using. Skills that need a different env (e.g., LLM SFT needs ≥40 GB but local is MPS) consult the list, propose the smallest sufficient env, and ask the user to confirm the switch (cost implication). The user can also manually switch at any time.

`env.json` is refreshed only on explicit invocation of `dl-detect-env` or on `dl-remote-execute` failure.

## Remote execution dispatcher

`dl-remote-execute` is the most complex new skill.

### Contract

**Input** (passed by orchestrator):
- Path to script in workdir.
- Estimated resource requirement (VRAM, walltime, disk, network egress).
- Target environment name (from `env.json`).
- Files to upload (dataset, checkpoints, configs).
- Files to fetch back (model artifacts, logs, charts, metrics).

**Output** (returned to orchestrator):
- Exit code (mirrors local execution).
- Stdout / stderr captured to `<workdir>/remote_logs/<env>_<step>.log`.
- Fetched artifacts placed back in workdir at expected paths.
- Cost estimate (best-effort, from provider response or rate card).
- Pointer the user can use to resume / inspect (Modal app URL, RunPod pod ID, SSH session marker).

### Per-provider sub-modes

| Provider | Sub-mode | Mechanism | Notes |
|---|---|---|---|
| Modal | `modal` | `modal run script.py` with auto-generated `@app.function` wrapper | Best DX. Encode known late-2025 GPU-sandbox-with-Volume crash workaround. |
| RunPod | `runpod` | `runpodctl create pod` → `ssh exec` → `runpodctl stop pod` | Pod lifecycle managed; warn on idle billing. |
| Vast.ai | `vastai` | `vastai search` / `create instance` → SSH → `destroy instance` | Cheapest; volatile; default to single-shot, never sustained. |
| Lambda Labs | `lambda` | API reserve → SSH (PyTorch preinstalled) | Slow to provision; good for long jobs. |
| Beam | `beam` | `beam run` with `@function` decorator | 2-3s cold starts; Modal alternative. |
| Generic SSH | `ssh-generic` | rsync up → ssh exec → rsync back | Universal escape hatch. |
| Colab | `colab-handoff` | Generates `.ipynb`, prints "open this URL, run all" instructions, polls for result file | Async; orchestrator parks the step until user confirms. |

### Decision flow ("ask once, continue silently")

```
1. Read env.json. List environments where kind == "remote" with auth ok, plus local if it fits.
2. Filter by VRAM requirement.
3. First remote step of a task → show top 3 candidates with cost + latency tradeoffs:
     1. modal       T4 → A10G    ~$0.79/hr   ~3s cold start    pay-per-second
     2. runpod-h100 H100 (warm)  ~$2.49/hr   ~10s SSH connect  pod stays alive
     3. ssh-gpu-box local 4090   $0          immediate         (your own box)
   User picks. Pin to env.json `active`.
4. Subsequent remote steps → continue using `active` silently.
5. Switch trigger: user explicitly says "switch to X", OR step's resource estimate exceeds `active` env (then re-prompt with new candidate set).
6. If no remote satisfies, surface "this needs N GB VRAM, none of your environments have it. Configure a larger one or reduce model size."
```

### Hard rules

- Never auto-launch paid compute without user OK on the first remote step of a session.
- Stream logs back during execution; do not make the user wait until job ends.
- On failure, fetch what is salvageable (last checkpoint, last log) before tearing down the remote.
- Always tear down ephemeral resources (Vast / RunPod pods) on completion, success or failure. Surface cost actually incurred.
- Never leave secrets in the script that gets uploaded — secrets stay in env vars set on the remote, not in the file body.

## Skill structure & content rules

### File structure

```
skills/dl-<name>/
├── SKILL.md          (always present)
└── scripts/          (optional — only if the skill ships executable helpers)
```

### `SKILL.md` frontmatter (mandatory)

```yaml
---
name: dl-<name>
description: Use when <trigger>. Do NOT use when <anti-trigger>.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---
```

### Body structure

Each skill body has these sections, scaled to actual content (no fixed line cap):

1. **One-line purpose.** What this skill produces.
2. **When to invoke.** Trigger conditions, copied/expanded from frontmatter.
3. **When NOT to invoke.** Anti-triggers (prevents over-firing).
4. **Decision rules.** If the skill picks among options (e.g., `dl-llm-pref-opt` picks DPO/KTO/ORPO/GRPO), encode the decision tree.
5. **Recipe template.** Skeleton script the orchestrator adapts. Always references workdir-relative paths.
6. **Research hooks.** Explicit lines like *"Before writing code, invoke `ml-engineer-research` for: current `target_modules` recommendation for {model_family}."* This is how skills stay fresh.
7. **Verification gates.** What `ml-engineer-verify` should check after this skill runs.
8. **Hard constraints.** Skill-specific rules — what NOT to do.
9. **Output checklist.** Self-check the skill runs before returning.

### Content rules across all 29 skills

- **No baked-in version numbers.** "Use the latest stable HF transformers" — never "use transformers==4.46". Version pinning is the user's venv concern.
- **No baked-in benchmark numbers.** VRAM math gets stale. Defer to `dl-detect-env` and runtime probes.
- **Opinionated defaults with override path.** Pattern: *"Default: Unsloth (fastest single-GPU). Override path: if user has multi-GPU or names Axolotl/TRL, switch."* User flexibility is preserved at every decision point.
- **Research-delegated facts.** Anything that changes month-to-month → research hook, not encoded.
- **Cross-references.** Skills can name other skills they depend on / hand off to. Orchestrator reads these references when planning the loop.
- **No code in `SKILL.md` body that runs on its own.** Code blocks are templates the orchestrator adapts via `dl-finetune-loop` / `ml-engineer-write-code`.

### Naming consistency

- Skill files: `dl-<domain>-<verb>` (`dl-cv-classify`, `dl-llm-pref-opt`).
- Cross-domain infra: `dl-<noun>` (`dl-detect-env`, `dl-checkpoint`).
- Domain prefixes are lowercase: `cv`, `nlp`, `llm`, `vlm`.

### Writing style — hybrid (caveman safe, prose where load-bearing)

| Section | Style |
|---|---|
| Frontmatter (`name`, `description`) | Terse imperative. Drop articles. Keep negations explicit ("Do NOT use when..."). |
| Section headers | Noun phrases, no articles ("When to invoke", "Decision rules"). |
| Step lists | Imperative bullets, no pleasantries ("Read env.json", not "You should read the env.json file"). |
| **Decision rules** | **Full prose with explicit IF/THEN/UNLESS, capitalized NOT/NEVER/MUST.** No compression. |
| **Error handling / hard constraints** | **Full prose. Explicit antecedents.** No pronouns whose referent is ambiguous. |
| Recipe templates (code skeletons) | Verbatim. Never compressed. |
| Verification gates | Imperative bullets, conditions stated explicitly ("Verify loss decreased monotonically over last 50 steps. If NOT, fail."). |
| Research hooks | Full prose stating the question to ask `ml-engineer-research`. |
| Output checklist | Imperative bullets. |

**Two anti-patterns forbidden in skill files:**

- **No bare negation drops.** "X must not be used" — never compress to "X used."
- **No conditional collapse.** "If A and not B then C" stays as full prose — never tabularized when it has nested logic.

### v1 acceptance criterion per skill

A skill is "v1 done" when:

- [ ] Frontmatter trigger is unambiguous (no other skill could reasonably claim the same trigger).
- [ ] One end-to-end recipe template the orchestrator can adapt without inventing steps.
- [ ] Verification gates are explicit (what `ml-engineer-verify` checks after this skill runs).
- [ ] Research hooks are present for any fact stale within 6 months.
- [ ] Hard constraints are listed in full prose with explicit negations.
- [ ] Hybrid writing style applied: terse headers / bullets, full prose for decisions and errors.

## Implementation phasing

### Phase 1 — Foundation (7 skills + router + 3 sub-agent shells)

- All 7 infra skills: `dl-detect-env`, `dl-remote-execute`, `dl-experiment-track`, `dl-checkpoint`, `dl-distributed`, `dl-debug-training`, `dl-prior-art`.
- Router prologue added to `ml-engineer.md`.
- Three sub-agent shells (`cv-engineer.md`, `nlp-engineer.md`, `llm-engineer.md`) — each has the loop documented but only references infra skills + 1 placeholder domain skill.
- Plugin version bump to `0.2.0-alpha.1`.

After Phase 1: the plugin can route a DL task to the right sub-agent, detect the environment, hand off to a remote provider, and run a generic finetune script. No domain-specific intelligence yet.

### Phase 2 — CV + NLP breadth (10 skills)

- CV: `dl-cv-classify`, `dl-cv-detect`, `dl-cv-segment`, `dl-cv-eval`.
- NLP: `dl-nlp-classify`, `dl-nlp-token`, `dl-nlp-eval`.
- Data/training core: `dl-load-data`, `dl-augment`, `dl-finetune-loop`.
- Plugin version bump to `0.2.0-alpha.2`.

After Phase 2: end-to-end CV and NLP tasks work. LLM/VLM tasks still route to LLM sub-agent but only get generic finetune support.

### Phase 3 — LLM + VLM + ensembling (12 skills)

- LLM: `dl-llm-lora`, `dl-llm-instruction-tune`, `dl-llm-pref-opt`, `dl-llm-eval`, `dl-llm-merge`, `dl-llm-quantize`, `dl-llm-serve`.
- VLM: `dl-vlm-finetune`.
- Cross-domain: `dl-pseudo-label`, `dl-distillation`, `dl-ensemble-tta`.
- Low-priority: `dl-cv-pretrain`.
- Plugin version bump to `0.2.0`.

After Phase 3: full taxonomy shipped. v1 complete.

## Success criteria for v1

The plugin is "v1 done" when, on a clean machine with the venv already set up:

- [ ] User says *"finetune ResNet-50 on CIFAR-10"* → router → cv-engineer → trains, verifies, reports. End-to-end in under 5 minutes on a small subset.
- [ ] User says *"NER on this CoNLL file"* → router → nlp-engineer → trains DeBERTa-v3-small token classifier, evaluates F1.
- [ ] User says *"QLoRA-finetune Qwen2.5-0.5B on this dataset"* → router → llm-engineer → uses Unsloth on local if VRAM fits, else surfaces remote candidates → trains, eval, saves merged model.
- [ ] User says *"merge these two LoRA adapters"* → llm-engineer → invokes `dl-llm-merge` → produces merged model with eval.
- [ ] On any task above, if user has only Mac MPS and the task needs more, `dl-detect-env` lists user's remotes and `dl-remote-execute` shows top 3 candidates (cost + latency tradeoffs visible).
- [ ] Every step ends with `ml-engineer-verify` returning `verified` before continuing.
- [ ] Every multi-step task ends with `ml-engineer-review` returning `release` or `release-with-caveats`.

## Risk register

| Risk | Mitigation |
|---|---|
| Router mis-classifies task domain | Router asks one clarifying question on ambiguous signal; user can also explicitly invoke a sub-agent. |
| Remote provider CLI breaks | `dl-detect-env` re-probes on demand; `dl-remote-execute` shows clear error if provider responds unexpectedly; user can skip to next candidate. |
| Skill descriptions over-fire (too broad triggers) | Explicit "Do NOT use when..." anti-triggers in every skill frontmatter. |
| Skill descriptions under-fire (too narrow triggers) | Phase rollouts let real usage surface gaps before v1 freezes. |
| LoRA / DPO / etc APIs change in HF libs | Research hooks delegate freshness to runtime; recipe templates kept minimal so HF API drift only invalidates the template, not the skill structure. |
| User runs up cloud bills accidentally | First remote step of a session always confirms; tear-down on completion is mandatory; cost actually incurred is surfaced after each remote run. |
| `env.json` goes stale during long task | User can manually invoke `dl-detect-env` to refresh; auto-refresh on remote-execute failure. |
| Hybrid writing style (compressed bullets) hurts skill triggering accuracy | Hybrid rule keeps decision rules, error handling, and negations in full prose; only headers, bullets, and pleasantries get compressed. Anti-pattern list explicitly forbids compressing nested conditionals or dropping NOT/NEVER/MUST. |

## Open questions deferred to implementation

- Exact cost rate cards per provider (rates change; encode lookup, not values).
- Whether `dl-experiment-track` should default to wandb or mlflow (defer to skill author; user override at any point).
- Whether `dl-llm-eval` should ship a built-in benchmark suite or always defer to lm-evaluation-harness defaults (lean toward defer).
- Exact router heuristics for ambiguous file types (e.g., `.parquet` of embeddings — tabular or NLP?).

## References

Sources for the taxonomy refinement (late 2025 / early 2026):

- Axolotl vs Unsloth vs TorchTune comparison — Spheron, 2026.
- Fine-Tuning in 2026: Axolotl vs Unsloth vs TRL vs LLaMA-Factory — DEV.to.
- Best frameworks for fine-tuning LLMs in 2025 — Modal blog.
- Hugging Face: Preference Tuning LLMs (DPO methods).
- Philipp Schmid: How to align open LLMs in 2025 with DPO.
- Axolotl RLHF docs.
- LLM Inference Servers Compared: vLLM vs TGI vs SGLang vs Triton (2026).
- vLLM vs SGLang — Techsy 2026.
- EleutherAI lm-evaluation-harness (GitHub).
- HF: Integrating benchmarks into LM Evaluation Harness.
- HF timm (GitHub).
- Ultralytics SAM 2 / SAM 3 docs.
- HF Cookbook: Fine-tuning Qwen2-VL with TRL.
- HF: Preference Optimization for VLMs.
- LLM Quantization Guide: GGUF vs AWQ vs GPTQ vs bitsandbytes (2026).
- vLLM Quantization Complete Guide — Jarvis Labs.
- HF Accelerate: FSDP vs DeepSpeed.
- HF: From DeepSpeed to FSDP and Back.
- State of ML Competitions 2025 — mlcontests.com.
- NVIDIA: Winning a Kaggle Competition with Generative AI–Assisted Coding (2026).
- NVIDIA Kaggle Grandmasters Playbook.
- Albumentations: Test-Time Augmentation.
- HF: Merge LLMs with mergekit (mlabonne).
- arcee-ai/mergekit (GitHub).
- HF PEFT (GitHub).
- NVIDIA: Introducing DoRA.
- QDoRA: The new PEFT standard for 2025.
- Modal Changelog and product updates (Sep / Oct 2025).
- GPU Cloud Showdown: Lambda vs CoreWeave vs RunPod vs Vast vs Modal — DEV.to.
- Beam Cloud: Top Serverless GPU Providers 2025.
- arXiv: Knowledge Distillation of LLMs survey 2025.

Caveman / compressed prompting research basis: LLMLingua (Jiang et al. 2023, arXiv:2310.05736); LLMLingua-2 (2024). Hybrid style adoption is a design choice, not benchmark-validated for skill files specifically.
