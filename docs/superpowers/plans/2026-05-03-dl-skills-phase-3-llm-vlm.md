# DL Skills Phase 3 — LLM + VLM + Ensembling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `ml-engineer` plugin v0.2.0 (no more alpha) — add the final 12 deep-learning skills covering LLM finetuning (7), VLM finetuning (1), and cross-domain training extras (4: pseudo-labeling, distillation, ensembling+TTA, CV self-supervised pretraining). After Phase 3, the full 33-skill taxonomy is complete and the plugin can take any DL task end-to-end.

**Architecture:** Pure markdown additions to a Claude Code plugin. Each skill is a self-contained `SKILL.md` with frontmatter trigger + body following the structure from Phases 1-2: When/When NOT/Decision rules/Process/Recipe template/Hard constraints/Research hooks/Verification gates/Output checklist. Hybrid writing style: terse headers/bullets, full prose for decisions and constraints.

**Tech Stack:** Markdown + YAML frontmatter. Recipe templates show Python with HuggingFace ecosystem (transformers, peft, trl, unsloth, axolotl, vllm, sglang, bitsandbytes, mergekit, lighteval, lm-evaluation-harness, llama.cpp/ggml).

**Reference docs:**
- Spec: [`docs/superpowers/specs/2026-05-01-dl-skills-design.md`](../specs/2026-05-01-dl-skills-design.md)
- Phase 1 plan: [`docs/superpowers/plans/2026-05-02-dl-skills-phase-1-foundation.md`](2026-05-02-dl-skills-phase-1-foundation.md)
- Phase 2 plan: [`docs/superpowers/plans/2026-05-03-dl-skills-phase-2-cv-nlp.md`](2026-05-03-dl-skills-phase-2-cv-nlp.md)

**Lessons absorbed from Phase 1+2 reviews (encode in every Phase 3 skill upfront):**
- Recipe template + Research hooks sections per design spec.
- `WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()` idiom in any Python skeleton.
- Decision rules + Hard constraints in full prose with explicit IF/THEN/NEVER/MUST.
- No baked-in version pins or benchmark numbers as facts.
- Cross-references to sister skills with exact correct names.
- `weights_only=True` on any `torch.load()`.
- `subprocess.run(check=True)` not `os.system(...)`.
- Eval skills NEVER apply augmentation.
- Multi-domain skills with internal selectors (per Phase 3 brainstorm Q1, Q3, Q5) — not split into N sub-skills.

**Phase 3 brainstorm decisions (locked):**
- Q1: `dl-llm-eval` stays as ONE skill with internal modes (lm-eval-harness / lighteval / custom). Different from Phase 2 eval split — LLM eval libraries already abstract across benchmarks.
- Q2: `dl-llm-lora` defaults to Unsloth single-GPU, Axolotl multi-GPU. Per spec.
- Q3: `dl-llm-pref-opt` stays as ONE skill — DPO/KTO/ORPO/GRPO selector based on data shape.
- Q4: `dl-cv-pretrain` ships as a "recognize when to skip" skill with thin recipe — fuller content waits for v2.
- Q5: `dl-ensemble-tta` stays as ONE skill with sub-modes (CV TTA, NLP cross-fold blend, tabular OOF rank-average).

**Phase 3 scope:**
- LLM (7): `dl-llm-lora`, `dl-llm-instruction-tune`, `dl-llm-pref-opt`, `dl-llm-eval`, `dl-llm-merge`, `dl-llm-quantize`, `dl-llm-serve`.
- VLM (1): `dl-vlm-finetune`.
- Cross-domain (4): `dl-pseudo-label`, `dl-distillation`, `dl-ensemble-tta`, `dl-cv-pretrain`.

---

## File Structure

**New files (12 skill SKILL.mds):**

```
skills/
├── dl-llm-lora/SKILL.md
├── dl-llm-instruction-tune/SKILL.md
├── dl-llm-pref-opt/SKILL.md
├── dl-llm-eval/SKILL.md
├── dl-llm-merge/SKILL.md
├── dl-llm-quantize/SKILL.md
├── dl-llm-serve/SKILL.md
├── dl-vlm-finetune/SKILL.md
├── dl-pseudo-label/SKILL.md
├── dl-distillation/SKILL.md
├── dl-ensemble-tta/SKILL.md
└── dl-cv-pretrain/SKILL.md
```

**Modified files (5):**

```
.claude-plugin/plugin.json              (version 0.2.0-alpha.2 → 0.2.0)
agents/cv-engineer.md                   (mark Phase 3 skills as available)
agents/nlp-engineer.md                  (mark Phase 3 skills as available)
agents/llm-engineer.md                  (mark all LLM/VLM Phase 3 skills as available)
README.md                               (Phase 3 status section + directory tree update)
```

**Implementation order:**

Phase 3 skills are mostly independent (no `data_policy.json`-style cross-skill data contracts beyond the existing one). Most can run in parallel after a small batch of "foundations":

1. **Foundation order (sequential)** — Tasks 1-2:
   - `dl-llm-lora` (lots of skills depend on its decisions: Unsloth-vs-Axolotl, target_modules, etc.).
   - `dl-llm-instruction-tune` (depends on lora; absorbs SFT data formatting).
2. **Parallel wave** — Tasks 3-13 (10 skills written in parallel + commit sequentially):
   - `dl-llm-pref-opt`, `dl-llm-eval`, `dl-llm-merge`, `dl-llm-quantize`, `dl-llm-serve`
   - `dl-vlm-finetune`
   - `dl-pseudo-label`, `dl-distillation`, `dl-ensemble-tta`, `dl-cv-pretrain`
3. **Wire-up + finalization** — Tasks 14-17:
   - Sub-agent updates.
   - Plugin version bump to 0.2.0 (no alpha).
   - README update.
   - Cross-validation pass.
4. **Final review + push** — Task 18.

---

## Pre-flight

- [ ] **Step 0.1: Confirm branch + working tree**

Run: `cd /Users/lokesh/Desktop/code/Bio_hacking/ML_Engineer && git status && git branch --show-current`
Expected: clean tree on `phase-3-llm-vlm`, base is `bac25c2` (Phase 2 final polish) or later.

- [ ] **Step 0.2: Verify Phase 1+2 deliverables present**

Run: `ls skills/dl-* | wc -l` (counts directory entries; with -d this would be cleaner but shell varies — use `ls -d skills/dl-* | wc -l` or count via `find`).
Expected: 21 directories.

---

## Tasks 1-2 (sequential foundation)

### Task 1: dl-llm-lora — PEFT/LoRA/QLoRA/DoRA decision tree, Unsloth single-GPU default

The orchestrator subagent will write the full SKILL.md following the established pattern. Content scope:

- **Frontmatter description**: triggers on LoRA/QLoRA/DoRA mentions or any LLM finetune where parameter-efficient is appropriate. Anti-trigger: full finetune (use a future skill if added) or non-LLM tasks.
- **Decision rules**: pick LoRA / QLoRA (4-bit, default for memory-constrained) / DoRA (incremental winner per spec) / full-finetune. Pick library: Unsloth single-GPU default (2-5x faster, 80% less memory per spec, 3 Kaggle wins 2024-2025); Axolotl multi-GPU; TRL when user names it. User overrides preserved.
- **Process Steps**: consult dl-prior-art → check VRAM via env.json → pick method+library → set target_modules (research-hooked) → wire training.
- **Recipe template**: Unsloth `FastLanguageModel.from_pretrained(load_in_4bit=True)` + `FastLanguageModel.get_peft_model(...)`. Axolotl yaml config. TRL `LoraConfig` example.
- **Hard constraints**: NEVER mix Unsloth with FSDP/DeepSpeed (already in dl-distributed). NEVER set r > 64 without justification. NEVER use bnb 4-bit quant for serving (training-only). NEVER use a tokenizer from a different model.
- **Research hooks**: current `target_modules` per model family (changes), DoRA vs LoRA effectiveness updates, Unsloth-supported model list (changes).
- **Verification gates**: trainable_params printed, base model frozen, adapter saved with `save_pretrained`, sanity generation matches chat template.

Commit: `Add dl-llm-lora skill: PEFT/LoRA/QLoRA/DoRA selector with Unsloth single-GPU default`

### Task 2: dl-llm-instruction-tune — SFT format conventions, chat templates, packing, response-only masking

Content scope:

- **Description**: triggers on SFT/instruction-tune/chat-finetune. Anti-trigger: preference tuning (use dl-llm-pref-opt) or pretraining.
- **Decision rules**: pick chat template (read from base model's tokenizer or override). Decide on packing (default: enabled for Unsloth, disabled for short-format datasets). Decide on response-only masking (mandatory if SFT data has long instruction prefixes).
- **Process**: consult prior-art → format data per chat template → verify with 5 decoded examples → wire SFTTrainer (TRL) or Unsloth equivalent → smoke test (10 steps + 1 generation).
- **Recipe template**: TRL `SFTTrainer` example with `dataset_text_field`, `formatting_func`, `packing`, `response_template` for masking. Unsloth equivalent.
- **Hard constraints**: chat template MUST match base model. ALWAYS run a generation sanity check (loss alone proves nothing; broken format produces fluent garbage at low loss). NEVER train on data that mixes templates. NEVER skip response-only masking on long-instruction data — model wastes capacity reproducing instructions.
- **Research hooks**: current TRL SFTTrainer API surface, current chat template registry per model family.
- **Verification gates**: 5 formatted training examples saved + decoded for inspection. Generation sample after training included in <workdir>/samples/. Loss decreased over smoke-test 10 steps.

Commit: `Add dl-llm-instruction-tune skill: SFT with chat templates, packing, response-only masking`

---

## Tasks 3-13 (parallel wave — 10 skills written in parallel, committed sequentially)

For each task below, the orchestrator dispatches a parallel implementer with the full SKILL.md content embedded in the dispatch prompt (same pattern as Phase 2 Tasks 4-14). Implementers WRITE files but do NOT commit; controller commits each in a sequential pass.

### Task 3: dl-llm-pref-opt — DPO/KTO/ORPO/GRPO selector

- **Description**: triggers on preference-tuning / RLHF / DPO/KTO/ORPO/GRPO. Anti-trigger: SFT (use dl-llm-instruction-tune) or pretraining.
- **Decision rules**: pairwise (chosen/rejected pairs) → DPO. Binary thumbs-up/down → KTO. Single-GPU memory-tight → ORPO (combines SFT + preference in one pass). Verifiable reward (math, code, structured output) → GRPO.
- **Recipe template**: TRL `DPOTrainer`, `KTOTrainer`, `ORPOTrainer`, `GRPOTrainer` skeletons.
- **Hard constraints**: NEVER skip the SFT step before DPO unless using ORPO (which combines them). NEVER use a reward model trained on a different prompt template. NEVER mix preference data formats — pick chosen/rejected vs binary and stick.

Commit: `Add dl-llm-pref-opt skill: DPO/KTO/ORPO/GRPO selector keyed to data shape`

### Task 4: dl-llm-eval — lm-evaluation-harness + lighteval with vLLM/SGLang backends

- **Description**: triggers on LLM evaluation / benchmark / leaderboard. Anti-trigger: small custom NLP eval (use dl-nlp-eval-* from Phase 2).
- **Decision rules**: pick lm-eval-harness (academic benchmarks: MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, BBH) vs lighteval (HF leaderboard subset, faster) vs custom (user-provided rubric). Pick backend: vLLM (default, fast) vs SGLang (RAG/agents).
- **Recipe template**: lm-eval-harness CLI with `--model vllm`. lighteval Python API with vllm backend.
- **Hard constraints**: NEVER report eval scores without naming the exact benchmark version (MMLU has multiple variants). NEVER eval on training data. ALWAYS pin the backend version when reporting numbers (vllm/sglang versions affect generation determinism). NEVER cherry-pick benchmarks — report a small fixed suite.

Commit: `Add dl-llm-eval skill: lm-eval-harness + lighteval with vLLM/SGLang backends`

### Task 5: dl-llm-merge — mergekit (SLERP / TIES / DARE-TIES)

- **Description**: triggers on merge / ensemble of fine-tuned LLMs. Cheap and often top-of-leaderboard per spec.
- **Decision rules**: SLERP for 2-model interpolation. TIES for 3+ models with task arithmetic. DARE-TIES adds dropout for further generalization. User picks via prior-art recommendation.
- **Recipe template**: mergekit yaml configs for SLERP / TIES / DARE-TIES, plus the `mergekit-yaml` CLI invocation.
- **Hard constraints**: NEVER merge models with different tokenizers. NEVER merge models with different architectures. ALWAYS evaluate the merge with `dl-llm-eval` after — merging can silently degrade if weights conflict.

Commit: `Add dl-llm-merge skill: mergekit SLERP/TIES/DARE-TIES selector`

### Task 6: dl-llm-quantize — Post-training quantization for serving

- **Description**: triggers on quantize / AWQ / GPTQ / GGUF / 4-bit / serving. Distinct from QLoRA's training-time bnb.
- **Decision rules**: AWQ default (fast, accurate, vLLM/SGLang support). GPTQ alt (older, broader hardware). GGUF for llama.cpp / local CPU+GPU inference. bitsandbytes for in-memory load (slower serving than AWQ).
- **Recipe template**: AWQ via `autoawq`. GPTQ via `auto_gptq`. GGUF via `llama.cpp` `convert.py` + quantize binary.
- **Hard constraints**: NEVER quantize without a representative calibration dataset (random text degrades AWQ/GPTQ quality). NEVER serve a bnb-4bit model in production — it's training-only. ALWAYS eval the quantized model with `dl-llm-eval` to measure degradation.

Commit: `Add dl-llm-quantize skill: AWQ/GPTQ/GGUF for serving with calibration discipline`

### Task 7: dl-llm-serve — vLLM + SGLang for eval-time inference

- **Description**: triggers on serve / vLLM / SGLang for benchmarking or generating synthetic data. Anti-trigger: production serving (out of scope per spec).
- **Decision rules**: vLLM default (high throughput, broad model support). SGLang for RAG / agents / multi-turn (RadixAttention prefix sharing).
- **Recipe template**: vLLM `LLM(model=...)` + `SamplingParams`. SGLang `srt.set_default_backend(...)` + function calls.
- **Hard constraints**: NEVER run vLLM on a model with `tokenizer.padding_side="left"` set wrong for the model architecture. ALWAYS warm the engine before measuring throughput. NEVER assume vLLM and HF generate identical outputs — vLLM uses different sampling internals; pin generation params explicitly.

Commit: `Add dl-llm-serve skill: vLLM/SGLang for eval-time and synthetic-data generation`

### Task 8: dl-vlm-finetune — Qwen2-VL / Pixtral / LLaVA / SmolVLM2 finetune

- **Description**: triggers on VLM / vision-language / Qwen-VL / Pixtral / LLaVA / SmolVLM. Substitutes for dl-llm-instruction-tune at training step.
- **Decision rules**: pick VLM via prior-art + user confirm. Default suggestions: Qwen2.5-VL (general), Pixtral (image-text reasoning), LLaVA (research baseline), SmolVLM2 (small / edge).
- **Recipe template**: TRL Qwen2-VL cookbook adaptation + Axolotl Qwen2.5-VL/Pixtral/LLaVA/SmolVLM2 yaml.
- **Hard constraints**: image preprocessing MUST match base model (resolution, channel order, pixel range). NEVER mix image tokenization across VLM families. ALWAYS run a generation sanity check on held-out images.

Commit: `Add dl-vlm-finetune skill: TRL+Axolotl recipes for Qwen-VL/Pixtral/LLaVA/SmolVLM`

### Task 9: dl-pseudo-label — Confidence-thresholded self-training

- **Description**: triggers on pseudo-label / self-training / semi-supervised. Recurring Kaggle staple.
- **Decision rules**: confidence threshold (default 0.9 for hard labels, raise if pseudo-label quality is low). K-fold-safe: generate pseudo-labels with held-out folds, never with the model that uses them. Distill into single model after.
- **Recipe template**: HF Dataset pseudo-label pipeline with confidence filtering, plus a distillation handoff to dl-distillation.
- **Hard constraints**: NEVER pseudo-label and train with the same fold's predictions. NEVER lower the confidence threshold below the model's calibration ECE. ALWAYS verify pseudo-labels are non-trivially better than training-only baseline before adopting.

Commit: `Add dl-pseudo-label skill: confidence-thresholded self-training with k-fold safety`

### Task 10: dl-distillation — Logit / feature / CoT distillation

- **Description**: triggers on distill / teacher-student / model compression / CoT distillation.
- **Decision rules**: logit distillation (KL divergence on softmax) for classification. Feature distillation (MSE on hidden states) for representation transfer. CoT distillation (teacher generates reasoning chains, student trained on Q→reasoning→A) for LLM reasoning.
- **Recipe template**: HF logit distillation (custom loss in Trainer). CoT distillation pipeline using `dl-llm-serve` for teacher generation + `dl-llm-instruction-tune` for student training.
- **Hard constraints**: NEVER distill from a teacher worse than the student baseline. NEVER use a temperature outside [1, 10] for logit distillation without research backing. ALWAYS evaluate distilled model against the original baseline.

Commit: `Add dl-distillation skill: logit/feature/CoT distillation with teacher-quality gate`

### Task 11: dl-ensemble-tta — K-fold OOF blend, rank-average, snapshot, TTA

- **Description**: triggers on ensemble / TTA / blend / OOF. Cross-domain (CV+NLP+tabular).
- **Decision rules**: simple-average (start). Rank-average (when confidences differ in scale). Weighted average (when base models differ in quality). Stacking (when 5+ models). TTA (test-time augmentation) for CV.
- **Recipe template**: per-domain helpers — CV TTA via dl-augment's recipes inverted at inference time; NLP cross-fold blend; tabular OOF rank-average.
- **Hard constraints**: NEVER blend predictions before applying the same calibration. NEVER stack on the same folds used for training (overfit). ALWAYS sanity-check ensemble vs best single model — ensembling often hurts when one model dominates.

Commit: `Add dl-ensemble-tta skill: cross-domain OOF blend / rank-average / snapshot / TTA`

### Task 12: dl-cv-pretrain — SimCLR / DINO / MAE (recognize-when-to-skip)

- **Description**: triggers on self-supervised CV pretraining / SimCLR / DINO / MAE. Spec calls out: skill's primary job is recognizing when to SKIP — most users should use timm pretrained.
- **Decision rules**: SKIP if pretrained timm backbones transfer to user's domain. CONSIDER pretraining if domain has lots of unlabeled images AND pretrained backbones underperform (medical, satellite, microscopy). DINO default (best across most settings). SimCLR for contrastive setup. MAE for masked-autoencoder reconstruction.
- **Recipe template**: thin — pointer to lightly / dino-vits, plus a one-screen DINO config.
- **Hard constraints**: NEVER pretrain when timm pretrained beats your domain — wasted compute. NEVER pretrain without confirming the unlabeled data is large enough (≥100k images typically).

Commit: `Add dl-cv-pretrain skill: DINO/SimCLR/MAE with skip-by-default discipline`

---

## Tasks 14-17 (finalization — sequential)

### Task 14: Sub-agent updates for Phase 3 availability

- **Files**: `agents/cv-engineer.md`, `agents/nlp-engineer.md`, `agents/llm-engineer.md`.
- For each: update the Phase 2 status section to add a Phase 3 status section listing the now-available skills. Remove `(Phase 3)` annotations from the skills tables for newly-shipped skills.
- LLM agent gets the most updates (7 LLM + 1 VLM = 8 newly available).
- CV/NLP agents get the cross-domain extras (`dl-pseudo-label`, `dl-distillation`, `dl-ensemble-tta`, `dl-cv-pretrain` for CV).

Commit: `Sub-agents: mark Phase 3 LLM/VLM/cross-domain skills as available`

### Task 15: Bump plugin version to 0.2.0 (no alpha)

- File: `.claude-plugin/plugin.json`.
- Version: `"0.2.0-alpha.2"` → `"0.2.0"`.
- This is the final v1 release per spec.

Commit: `Bump plugin version to 0.2.0 — full v1 release with all 33 skills shipped`

### Task 16: README update for Phase 3

- File: `README.md`.
- Update directory tree to include all 12 new skills (total dl-*: 33).
- Update header from "Phase 1 + Phase 2" to "v0.2.0 — full v1 release".
- Add Phase 3 status section listing the 12 new skills with one-line descriptions.
- Update the "Phase 3 deferred" language elsewhere in the README to reflect "all phases shipped".

Commit: `README: v0.2.0 release notes with Phase 3 status and full 33-skill directory tree`

### Task 17: Cross-validation pass

Same pattern as Phase 1 Task 14 / Phase 2 Task 18:
- All 12 new skill files present + frontmatter parses.
- Total dl-* skills count = 33.
- Plugin version = 0.2.0.
- Sub-agent updates correct.
- Cross-references resolve (no Phase 3 forward references should remain).
- No baked-in version pins.
- Working tree clean.

---

### Task 18: Phase 3 completion review + push to remote

- End-of-phase code review by superpowers:code-reviewer.
- Verdict must be `release` or `release-with-caveats`.
- Surgical fixes for any Important issues.
- Push `phase-3-llm-vlm` to remote.

---

## Acceptance criteria for Phase 3 (= v1 of plugin)

After Phase 3, the plugin is "v1 done" per the spec's success criteria:

- [ ] User says *"finetune ResNet-50 on CIFAR-10"* → cv-engineer → end-to-end (Phase 2 already verified).
- [ ] User says *"NER on this CoNLL file"* → nlp-engineer → end-to-end (Phase 2 already verified).
- [ ] User says *"QLoRA-finetune Qwen2.5-0.5B on this dataset"* → llm-engineer → uses Unsloth on local if VRAM fits, else surfaces remote candidates → trains, eval, saves merged model.
- [ ] User says *"merge these two LoRA adapters"* → llm-engineer → invokes `dl-llm-merge` → produces merged model with eval.
- [ ] On any task, if local doesn't fit, `dl-detect-env` lists remotes and `dl-remote-execute` shows top 3 candidates.
- [ ] Every step ends with `ml-engineer-verify` returning `verified` before continuing.
- [ ] Every multi-step task ends with `ml-engineer-review` returning `release` or `release-with-caveats`.

After Phase 3, ALL 33 skills shipped. Plugin reaches v0.2.0 (no more alpha).
