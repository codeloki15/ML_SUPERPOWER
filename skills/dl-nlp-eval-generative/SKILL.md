---
name: dl-nlp-eval-generative
description: Use to evaluate generative NLP outputs (summarization, translation, paraphrase, generative QA) — ROUGE-1/2/L, BLEU, BERTScore, perplexity. Pairs with seq2seq finetuning. Do NOT use for classification (use dl-nlp-eval-classify), token classification (use dl-nlp-eval-token), or LLM benchmark evaluation (use dl-llm-eval, Phase 3).
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# NLP Eval — Generative

Evaluate generated text against reference text. Compute n-gram metrics (ROUGE, BLEU), embedding-based metrics (BERTScore), and language-model perplexity. Surface concrete examples of best and worst outputs for human review — generative metrics are noisy; the human eyeball is the final judge.

## When to invoke

- After a generative model produces outputs on an eval set (summarization, translation, paraphrase, generative QA, dialogue response).
- User asks "evaluate this generation" / "what's the ROUGE / BLEU / BERTScore".

## When NOT to invoke

- Classification (use `dl-nlp-eval-classify`).
- Token classification (use `dl-nlp-eval-token`).
- LLM-on-benchmarks evaluation like MMLU / HellaSwag (use `dl-llm-eval`, Phase 3).
- Quality evaluation that needs LLM-as-judge — out of Phase 2 scope.

## Decision rules

### Metric set

- **Summarization**: ROUGE-1, ROUGE-2, ROUGE-L (recall-oriented). Add BERTScore for semantic similarity. Optionally perplexity of generations under a reference LM.
- **Translation**: BLEU (corpus-level, not sentence-level — sentence BLEU is noise). Add chrF or chrF++ as a secondary. BERTScore optional.
- **Paraphrase / generative QA**: BERTScore + exact-match rate.
- **Open-ended generation (no single reference)**: perplexity of model's outputs under a stronger reference LM, OR human review.

### Reference handling

- Multiple references per source: pass as list-of-lists. ROUGE / BLEU support multiple references natively.
- Single reference: standard.

### Decoding

- Decoding strategy AFFECTS evaluation. State which (greedy / beam / sampling) was used; metric scores are NOT comparable across strategies.

## Process

### Step 1 — Read generations + references

Read `<workdir>/predictions/generations.jsonl` (one `{source, prediction, reference}` per line). If references are multiple, format as list.

### Step 2 — Compute metrics

Save to `<workdir>/metrics.json`:

```json
{
  "rouge_1": 0.XXX,
  "rouge_2": 0.XXX,
  "rouge_L": 0.XXX,
  "bleu": 0.XXX,
  "chrf": 0.XXX,
  "bertscore_f1": 0.XXX,
  "perplexity": 0.XXX,
  "exact_match": 0.XXX,
  "best_examples": [...],
  "worst_examples": [...]
}
```

### Step 3 — Save best/worst examples

Pick top-5 and bottom-5 examples by aggregate score (e.g., ROUGE-L). Save to `<workdir>/predictions/best_worst_examples.txt` for human inspection.

### Step 4 — Surface insights

Print headline metric (ROUGE-L for summarization, BLEU for translation, BERTScore for paraphrase). Note the spread (top-5 vs bottom-5 score gap). Flag if generations are abnormally short / long compared to references (a common failure mode).

## Recipe template

### `<workdir>/src/_eval_nlp_generative.py`

```python
"""Generative NLP evaluation harness."""
import json
import os
from pathlib import Path

import numpy as np

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L (mid F1)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge_1": [], "rouge_2": [], "rouge_L": []}
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        scores["rouge_1"].append(s["rouge1"].fmeasure)
        scores["rouge_2"].append(s["rouge2"].fmeasure)
        scores["rouge_L"].append(s["rougeL"].fmeasure)
    return {k: float(np.mean(v)) for k, v in scores.items()}


def compute_bleu(predictions: list[str], references: list[list[str]]) -> dict:
    """Corpus BLEU. references: list-of-lists (one or more refs per pred)."""
    try:
        import sacrebleu
    except ImportError:
        return {"bleu": float("nan"), "chrf": float("nan")}
    refs_transposed = list(zip(*references)) if references and isinstance(references[0], list) else [references]
    bleu = sacrebleu.corpus_bleu(predictions, refs_transposed)
    chrf = sacrebleu.corpus_chrf(predictions, refs_transposed)
    return {"bleu": float(bleu.score), "chrf": float(chrf.score)}


def compute_bertscore(predictions: list[str], references: list[str], lang: str = "en") -> dict:
    """BERTScore F1 averaged over the eval set."""
    try:
        from bert_score import score
    except ImportError:
        return {"bertscore_f1": float("nan")}
    P, R, F = score(predictions, references, lang=lang, verbose=False)
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F.mean()),
    }


def compute_perplexity(model_id: str, texts: list[str], batch_size: int = 8) -> float:
    """Perplexity of `texts` under `model_id` (e.g., 'gpt2' as a reference LM)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    nlls = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        nlls.append(float(out.loss) * enc["input_ids"].shape[1])
    return float(np.exp(sum(nlls) / sum(enc["input_ids"].shape[1] for _ in nlls)))


def best_worst_examples(predictions: list[str], references: list[str],
                          rouge_l_scores: list[float], n: int = 5) -> dict:
    """Pick top-n and bottom-n by ROUGE-L."""
    indexed = list(enumerate(rouge_l_scores))
    indexed.sort(key=lambda x: x[1])
    worst = [{"pred": predictions[i], "ref": references[i], "rouge_L": s} for i, s in indexed[:n]]
    best = [{"pred": predictions[i], "ref": references[i], "rouge_L": s} for i, s in indexed[-n:]]
    return {"best_examples": best, "worst_examples": worst}


def save_metrics(metrics: dict):
    p = WORKDIR / "metrics.json"
    p.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Metrics saved to {p}")


def length_stats(predictions: list[str], references: list[str]) -> dict:
    """Predictions abnormally short or long is a common failure mode."""
    pred_lens = [len(p.split()) for p in predictions]
    ref_lens = [len(r.split()) for r in references]
    return {
        "pred_len_mean": float(np.mean(pred_lens)),
        "pred_len_p50": float(np.percentile(pred_lens, 50)),
        "ref_len_mean": float(np.mean(ref_lens)),
        "len_ratio_pred_to_ref": float(np.mean(pred_lens) / max(np.mean(ref_lens), 1)),
    }
```

## Hard constraints

- NEVER report sentence-BLEU. Use corpus-BLEU (`sacrebleu.corpus_bleu`) — sentence BLEU is dominated by length penalties on short sentences and is uninformative.
- NEVER compare scores across decoding strategies (greedy vs beam vs sampling) without disclaiming. Same model + same data, different decoding → different scores.
- NEVER report ROUGE alone for summarization without surfacing the length stats. ROUGE rewards longer outputs (more n-gram overlap chances).
- NEVER apply augmentation to the eval dataloader.
- NEVER skip the best/worst-examples surfacing. Generative metrics are noisy; humans need the qualitative spread.
- NEVER use BERTScore without specifying the language and the embedding model. Defaults vary across `bert_score` releases.
- NEVER cache generations across model versions.

## Research hooks

- **Reference LM choice for perplexity.** Query: *"Current recommended reference language model for perplexity-based evaluation of `{language}` generation as of {today}."*
- **BERTScore embedding model selection.** Query: *"Current best `bert_score` embedding model for `{language}` `{task_type}` evaluation as of {today}."*
- **Generative metric correlations with human judgment.** Query: *"Latest meta-evaluation of generative NLP metrics (ROUGE, BLEU, BERTScore, COMET, BLEURT) vs human judgment for `{task_type}` as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/metrics.json` exists with at minimum `rouge_L` (or `bleu` for translation) and `bertscore_f1`.
- `<workdir>/predictions/best_worst_examples.txt` exists with top-5 and bottom-5 examples surfaced.
- Length stats (`pred_len_mean`, `ref_len_mean`, `len_ratio`) included; flagged if `len_ratio < 0.5` or `> 2.0`.
- Decoding strategy (greedy / beam / sampling) is recorded in metrics or in a sibling `generation_config.json`.
- For BLEU: `corpus_bleu` was used, not sentence-level.
- Eval dataloader had NO augmentation.

## Output checklist

- [ ] Predictions and references loaded from generations.jsonl
- [ ] ROUGE / BLEU / BERTScore computed per task type
- [ ] Length stats reported
- [ ] Best 5 + worst 5 examples surfaced for human review
- [ ] Decoding strategy recorded
- [ ] Metrics JSON written
- [ ] Eval was NOT augmented
