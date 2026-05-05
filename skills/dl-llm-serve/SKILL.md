---
name: dl-llm-serve
description: Use to serve a finetuned LLM via vLLM (default, high throughput) or SGLang (RAG/agents/multi-turn with prefix caching) for eval-time inference, benchmarking, or synthetic-data generation. Scoped to "serve to benchmark / generate", NOT production serving (no autoscaling, monitoring, etc.). Do NOT use for production serving (out of v1 scope), in-process model.generate (use HF directly), or non-LLM models.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# LLM Serve (for eval / benchmarking / synthetic data)

Spin up vLLM or SGLang to serve an LLM for eval, benchmarking, or synthetic-data generation. NOT a production serving skill — no autoscaling, monitoring, request routing. Just "fire up vLLM, run a workload, tear down".

## When to invoke

- About to run `dl-llm-eval` and need a backend.
- Generating synthetic data via teacher LLM (for `dl-distillation` or data augmentation).
- Benchmarking inference throughput / latency for a finetuned model.
- User asks to "serve", "spin up vLLM/SGLang", "run inference at scale".

## When NOT to invoke

- Single in-process generation (use HF `model.generate(...)` — no need for a server).
- Production serving (autoscaling, monitoring, multi-tenant — out of v1 scope; use Modal/Replicate/Together as managed alternatives).
- Encoder NLP / VLM serving (different stacks).

## Decision rules

### Backend

- **vLLM (default)**: high throughput, broad model support, simplest to spin up. Use unless you need SGLang's specialty.
- **SGLang**: RadixAttention prefix caching — much faster on multi-turn, RAG, or agent workloads with shared system prompts. Use when prompts share long prefixes (10x+ speedup on those workloads).

### Mode

- **Engine-as-library** (use case: in-script eval / synthetic data): `from vllm import LLM, SamplingParams` then call `llm.generate(...)`. Lowest overhead.
- **OpenAI-compatible HTTP server** (use case: hand off to a separate process / external eval tool): `vllm serve <model>` exposes `/v1/chat/completions`. Use when downstream tool expects OpenAI API.

### Model loading

- For AWQ/GPTQ quantized: `LLM(model=..., quantization="awq")` or `quantization="gptq"`.
- For bf16/fp16 unquantized: `LLM(model=...)`.
- For LoRA adapter: `LLM(model=base, enable_lora=True)` then pass adapter at request time, OR merge first via `peft.merge_and_unload`.
- For GGUF: vLLM doesn't load GGUF — use `llama-cpp-python` instead.

## Process

### Step 1 — Pick backend per decision rules

Surface choice + rationale.

### Step 2 — Pick mode (library vs HTTP)

Surface choice; user can override.

### Step 3 — Build engine

Generate code. Pass appropriate `quantization`, `dtype`, `max_model_len`, `tensor_parallel_size`.

### Step 4 — Warm up

Run 5 dummy generations to warm caches BEFORE measuring throughput. Cold engine throughput is misleading.

### Step 5 — Run workload

Eval, batch generation, or HTTP server.

### Step 6 — Tear down

`del llm; torch.cuda.empty_cache()` (library mode) OR kill the server process (HTTP mode). vLLM/SGLang hold significant VRAM; release for downstream skills.

## Recipe template

### `<workdir>/src/_serve_vllm_lib.py` (engine-as-library)

```python
"""vLLM engine-as-library for batch generation / eval."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def build_vllm_engine(model_path: str, quantization: str | None = None,
                      max_model_len: int = 4096, tensor_parallel_size: int = 1,
                      dtype: str = "auto"):
    """quantization in {None (bf16/fp16), 'awq', 'gptq', 'fp8'}."""
    from vllm import LLM
    return LLM(
        model=model_path,
        quantization=quantization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=True,
    )


def make_sampling_params(temperature: float = 0.0, top_p: float = 1.0,
                          max_tokens: int = 512, stop: list[str] | None = None):
    from vllm import SamplingParams
    return SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop or [],
    )


def warm_up(llm, n: int = 5):
    """Run n dummy generations to warm caches before measuring throughput."""
    sampling = make_sampling_params(max_tokens=16)
    _ = llm.generate(["hello"] * n, sampling_params=sampling, use_tqdm=False)


def batch_generate(llm, prompts: list[str], sampling) -> list[str]:
    """Generate completions for a list of prompts."""
    outputs = llm.generate(prompts, sampling_params=sampling)
    return [o.outputs[0].text for o in outputs]


def teardown(llm):
    """Release VRAM. Important for downstream skills."""
    import gc, torch
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### `<workdir>/src/_serve_vllm_http.py` (OpenAI-compatible server)

```python
"""vLLM HTTP server. Use when downstream tool expects OpenAI API."""
import os
import subprocess
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def start_vllm_server(model_path: str, port: int = 8000, quantization: str | None = None,
                      tensor_parallel_size: int = 1, max_model_len: int = 4096) -> subprocess.Popen:
    """Spawn vllm serve in background. Returns the subprocess.Popen handle so caller can terminate."""
    cmd = [
        "vllm", "serve", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])
    log_path = WORKDIR / "vllm_server.log"
    log_file = log_path.open("w")
    print(f"Starting: {' '.join(cmd)} -> log {log_path}")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return proc


def stop_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
```

### `<workdir>/src/_serve_sglang.py` (SGLang for prefix-cached workloads)

```python
"""SGLang for RAG / agent workloads with shared prefixes."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def launch_sglang_runtime(model_path: str, port: int = 30000, dtype: str = "bfloat16"):
    """Use sglang's runtime; supports RadixAttention prefix caching."""
    import sglang as sgl
    runtime = sgl.Runtime(model_path=model_path, port=port, dtype=dtype)
    sgl.set_default_backend(runtime)
    return runtime


def shutdown_runtime(runtime):
    runtime.shutdown()
```

## Hard constraints

- ALWAYS warm the engine before measuring throughput. Cold-start latency is 5-30 seconds; first-batch metrics are misleading.
- NEVER assume vLLM and HF generate identical outputs. vLLM uses different sampling internals; pin all generation params (`temperature`, `top_p`, `seed`) explicitly.
- ALWAYS tear down vLLM/SGLang after use. They hold the model in VRAM until destroyed; downstream skills will OOM if you leave them running.
- NEVER serve a bitsandbytes-4bit model with vLLM. bnb is training-only; vLLM doesn't support it for serving. Use AWQ/GPTQ instead (`dl-llm-quantize`).
- NEVER use vLLM with `tokenizer.padding_side="left"` for chat templates — most chat templates assume right-padding for the prompt portion. Verify by inspecting tokenizer config before serving.
- NEVER expose the OpenAI-compatible HTTP server to a public network during eval / benchmarking — it has no auth.

## Research hooks

- **vLLM model support.** Query: *"Current vLLM supported model architectures and quantization types as of {today}."*
- **SGLang vs vLLM throughput on `{workload_type}`.** Query: *"Latest SGLang vs vLLM throughput comparison on `{workload_type}` (RAG / agent / single-turn) as of {today}."*
- **vLLM v1 engine status.** Query: *"vLLM V1 engine vs V0 — current default and any breaking API changes as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- Engine warmed up before any throughput measurement was reported.
- For library mode: `del llm + torch.cuda.empty_cache()` was called (or the script terminated).
- For HTTP server mode: the server process was terminated cleanly (or the user is keeping it intentionally for an external workload).
- Quantization arg matches the model's actual quantization (awq/gptq/none) — mismatch = silent garbage outputs.
- Generation params (temperature, top_p, seed) are explicit, not defaulted, when reproducibility matters.

## Output checklist

- [ ] Backend (vLLM / SGLang) chosen per decision rules
- [ ] Mode (library / HTTP) chosen per use case
- [ ] Engine built with correct quantization arg
- [ ] Warmed up (5 dummy generations) before measurement
- [ ] Workload ran (eval / batch generation / HTTP serve)
- [ ] Engine torn down; VRAM released
