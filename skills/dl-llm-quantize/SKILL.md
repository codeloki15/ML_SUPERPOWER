---
name: dl-llm-quantize
description: Use to quantize a finetuned LLM for serving — AWQ (default, fast+accurate, vLLM/SGLang support), GPTQ (alt, broader hardware), GGUF (llama.cpp/local CPU+GPU). Distinct from QLoRA's training-time bitsandbytes. Always re-eval after to measure degradation. Do NOT use for training (use dl-llm-lora QLoRA), pre-merge fp16/bf16 storage, or non-LLM models.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# LLM Quantize (for Serving)

Quantize an LLM for production / inference serving. Pick AWQ / GPTQ / GGUF based on serving target and hardware. Always calibrate with a representative dataset (random text degrades quality). Always evaluate post-quantization to measure degradation.

## When to invoke

- After SFT / pref-opt / merge, before deploying for inference.
- User asks to "quantize", "AWQ", "GPTQ", "GGUF", "4-bit for serving", "make it fit in less VRAM".

## When NOT to invoke

- Training-time quantization (use `dl-llm-lora` with QLoRA — bitsandbytes 4-bit).
- Saving full-precision checkpoints (use `dl-checkpoint`).
- Encoder NLP / VLM (different quantization stacks).

## Decision rules

### Method

- **AWQ (Activation-aware Weight Quantization, default)**: 4-bit weights, fast+accurate. Strong vLLM and SGLang support. Best speed-quality tradeoff for serving in 2024-2025+. Use unless a specific reason not to.
- **GPTQ**: older, broader hardware support (older CUDA). Use when target hardware doesn't support AWQ kernels.
- **GGUF (llama.cpp format)**: for CPU + GPU hybrid inference, edge / consumer hardware, llama.cpp ecosystem. Many bit-widths (Q4_K_M, Q5_K_M, Q8_0, etc.).
- **bitsandbytes 4-bit (NOT this skill's path)**: training-only; serving with bnb is slow. Use AWQ instead.

### Calibration dataset

REQUIRED for AWQ/GPTQ. Use a representative sample of the model's expected inputs:

- General chat: ~512 examples from C4, Pile, or instruct datasets.
- Domain-specific (medical, legal, code): use ~512 examples from the user's training distribution.
- NEVER use random text. Random calibration degrades AWQ/GPTQ quality 1-3%.

GGUF doesn't need calibration (uses round-to-nearest); just convert.

### Bit width

- AWQ: 4-bit (only option in canonical autoawq).
- GPTQ: 4-bit (most common); 3-bit / 8-bit possible but rarely worth the tradeoff.
- GGUF: Q4_K_M (recommended balance), Q5_K_M (slightly higher quality), Q8_0 (essentially lossless), Q2/Q3 (quality cliff — avoid for serving).

## Process

### Step 1 — Pick method + bit width per decision rules

Surface choice. User can override.

### Step 2 — Prepare calibration dataset

For AWQ/GPTQ: load 512 samples from a representative dataset.

### Step 3 — Run quantization

Use the appropriate library (`autoawq`, `auto_gptq`, `llama.cpp`).

### Step 4 — Verify load

Load the quantized model with the appropriate loader (vLLM for AWQ/GPTQ; llama.cpp for GGUF). Confirm it loads without error.

### Step 5 — Re-eval (MANDATORY)

Hand off to `dl-llm-eval` to measure post-quantization metric. Compare against pre-quantization metric. Surface degradation in markdown table.

## Recipe template

### `<workdir>/src/_quantize_awq.py`

```python
"""AWQ quantization via autoawq."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def quantize_awq(model_path: str, calib_dataset: str, output_path: str | None = None,
                 w_bit: int = 4, q_group_size: int = 128, n_calib_samples: int = 512):
    """Quantize a HF model to AWQ 4-bit.

    calib_dataset is REQUIRED — pass an HF dataset id (e.g., 'pileval' for general models)
    OR a local JSONL of representative inputs from the user's training distribution.
    Domain-specific models MUST use a domain-specific calib set; generic 'pileval' degrades
    quality 1-3% on specialty domains. See Hard constraints.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    output_path = output_path or str(WORKDIR / "quantized_awq")

    quant_config = {
        "zero_point": True,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": "GEMM",  # vLLM-friendly variant
    }

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=True, device_map="auto")

    print(f"Quantizing with calib dataset={calib_dataset}, n_samples={n_calib_samples}")
    model.quantize(tokenizer, quant_config=quant_config,
                   calib_data=calib_dataset, max_calib_samples=n_calib_samples)

    model.save_quantized(output_path, safetensors=True)
    tokenizer.save_pretrained(output_path)
    print(f"AWQ-quantized model saved to {output_path}")
    return output_path
```

### `<workdir>/src/_quantize_gptq.py`

```python
"""GPTQ quantization via auto-gptq (legacy hardware support)."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def quantize_gptq(model_path: str, output_path: str | None = None,
                  calib_examples: list[str] | None = None,
                  bits: int = 4, group_size: int = 128, desc_act: bool = False):
    """Quantize a HF model to GPTQ. calib_examples: list of representative texts (~512)."""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    output_path = output_path or str(WORKDIR / "quantized_gptq")

    quantize_config = BaseQuantizeConfig(
        bits=bits, group_size=group_size, desc_act=desc_act, sym=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)

    if not calib_examples:
        raise ValueError("GPTQ requires calibration examples; pass a list of ~512 representative texts")
    examples = [tokenizer(t, return_tensors="pt").input_ids for t in calib_examples]

    print(f"Quantizing GPTQ {bits}-bit with {len(examples)} calibration examples")
    model.quantize(examples)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path
```

### `<workdir>/src/_quantize_gguf.py`

```python
"""GGUF conversion via llama.cpp's convert.py + quantize binary."""
import os
import subprocess
import sys
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def convert_to_gguf(hf_model_path: str, output_dir: str | None = None,
                    quant_type: str = "Q4_K_M", llama_cpp_root: str | None = None):
    """Convert HF model to GGUF and quantize. Requires llama.cpp built locally."""
    output_dir = output_dir or str(WORKDIR / "quantized_gguf")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    llama_cpp = llama_cpp_root or os.environ.get("LLAMA_CPP_ROOT", "/usr/local/llama.cpp")
    convert_py = Path(llama_cpp) / "convert_hf_to_gguf.py"
    quantize_bin = Path(llama_cpp) / "build" / "bin" / "llama-quantize"

    if not convert_py.exists():
        raise FileNotFoundError(f"llama.cpp convert script not found at {convert_py}; set LLAMA_CPP_ROOT")
    if not quantize_bin.exists():
        raise FileNotFoundError(f"llama-quantize binary not found at {quantize_bin}; build llama.cpp")

    f16_path = Path(output_dir) / "model_f16.gguf"
    quant_path = Path(output_dir) / f"model_{quant_type}.gguf"

    subprocess.run([sys.executable, str(convert_py), hf_model_path, "--outfile", str(f16_path), "--outtype", "f16"],
                   check=True)
    subprocess.run([str(quantize_bin), str(f16_path), str(quant_path), quant_type], check=True)
    print(f"GGUF model saved to {quant_path}")
    return str(quant_path)
```

## Hard constraints

- NEVER quantize without a representative calibration dataset (AWQ / GPTQ). Random text degrades quality 1-3%.
- NEVER serve a bitsandbytes-4bit model in production. bnb 4-bit is training-only; serving with bnb is 3-5x slower than AWQ at similar quality.
- ALWAYS eval the quantized model with `dl-llm-eval`. Quantization sometimes degrades 5-15% on reasoning benchmarks (GSM8K, BBH); user must know.
- NEVER use Q2 or Q3 GGUF for serving. Quality cliff is real; Q4_K_M is the floor for "actually usable".
- NEVER quantize a merged model without first eval-ing the merge. Otherwise you can't tell if degradation is from the merge or the quantization.
- NEVER quantize with the wrong tokenizer. Tokenizer must match base; mismatch silently produces fluent garbage.

## Research hooks

- **AWQ vs GPTQ on current hardware.** Query: *"Latest AWQ vs GPTQ speed/quality benchmarks on `{gpu_class}` as of {today}."*
- **GGUF quant type recommendations.** Query: *"Current llama.cpp recommended GGUF quant type for `{use_case}` (chat / reasoning / coding) as of {today}."*
- **Calibration dataset best practices.** Query: *"Recommended calibration dataset (size, source) for AWQ/GPTQ quantization of `{model_family}` as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- Quantized model directory exists at the chosen output path.
- Model loads with the appropriate loader (vLLM `LLM(model=..., quantization='awq')` for AWQ; etc.).
- Calibration was used (for AWQ/GPTQ) — check the output directory's metadata.
- A re-eval was run via `dl-llm-eval`; degradation < 5% on the headline benchmark, OR user was informed of higher degradation.
- Tokenizer copied to output directory (most loaders need it next to the weights).

## Output checklist

- [ ] Method (AWQ / GPTQ / GGUF) chosen per decision rules
- [ ] Calibration dataset prepared (AWQ / GPTQ only)
- [ ] Quantization ran end-to-end
- [ ] Quantized model loads with target inference backend
- [ ] Re-eval via `dl-llm-eval`; degradation reported
- [ ] User informed if degradation > 5%
