# Speed Benchmarks: C++ MLX Engine vs Python MLX

Benchmarks run on the same Apple Silicon machine (M-series) with the same GPT-style model:

- 8 layers, 9 heads, d_model 576, head_dim 64
- SwiGLU MLP (hidden ~1536), 24k vocab
- 45.7M parameters, 74 parameter tensors
- Training: batch 32, sequence 256
- Inference: greedy decode, prompt "The future of AI is"

## Summary

| Workload | Python MLX | C++ MLX | Winner |
|----------|-------------|---------|--------|
| Training (Muon+AdamW) | 182 ms/step | 452 ms/step | Python 2.5x faster |
| Forward only (B=32, T=256) | 66 ms | 61 ms | C++ ~8% faster |
| Inference decode (KV cache, fp32) | 365 tok/s* | 621 tok/s | C++ 1.7x faster |
| Inference decode (KV cache, 4-bit) | — | 850 tok/s | C++ only |

*Python inference benchmark is full forward with no KV cache.

## Training

We measured the full training step with equivalent logic in both engines:

- Same model architecture
- Same Muon+AdamW optimizer (Newton-Schulz 5 steps for Muon)
- Same weight decay and bias correction
- `mx.compile()` in Python, `mlx::core::compile()` in C++

**Results:**
- **Python MLX:** 182 ms/step (5.5 steps/s)
- **C++ MLX:** 452 ms/step (2.2 steps/s)

Why Python is faster: Python's `mx.compile()` fuses the entire graph through `nn.value_and_grad`, including the backward pass. In C++ MLX, `value_and_grad` is a separate transform that `compile()` does not fully trace into, so the backward pass runs as many separate Metal kernels instead of one fused kernel. The C++ forward pass alone is actually slightly faster (61 ms vs 66 ms), but the backward pass is not fused as well.

What we tried to close the gap:

1. **Fused vg + optimizer** — compiled forward+backward+clip+optimizer in one function. Got 360 ms/step, still 2x slower than Python.
2. **Removed CPU syncs** — eliminated `.item<float>()` calls and per-step `eval()` for loss. Small gain.
3. **Tried shapeless compile** — failed with `Split cannot infer output shapes`.
4. **Reduced Newton-Schulz iterations from 5 to 3** — dropped to ~345 ms/step but validation loss regressed (1.216 vs 1.213).

Conclusion: C++ training is correct but not faster than Python for this graph. Reaching Python's speed would likely require either writing custom fused Metal kernels or an MLX C++ API change that makes `value_and_grad` fully traceable by `compile()`.

## Inference

| Variant | Prefill | Decode speed |
|---------|---------|--------------|
| Python MLX (full forward, no KV cache) | 4.2 ms (256 tokens) | 365 tok/s |
| C++ MLX (KV cache, fp32) | 8.6 ms | 621 tok/s |
| C++ MLX (KV cache, 4-bit) | 7.2 ms | 850 tok/s |

The C++ inference engine is faster because it:

1. **Uses a KV cache** — avoids recomputing past tokens each step.
2. **Uses fast MLX primitives** — `fast::rms_norm`, `fast::rope`, `fast::scaled_dot_product_attention`.
3. **Moves sampling to the GPU** — `argmax` and `categorical` stay on Metal, avoiding CPU sync.
4. **Supports 4-bit weight quantization** — `quantized_matmul` reduces memory bandwidth and matmul cost.

## Optimizations implemented

- **Inference engine** (`mlx-cpp/inference/`): checkpoint load, tokenizer via Python helper, KV cache, GPU sampling, optional 4-bit quantization.
- **Compiled SwiGLU** (`mlx-cpp/model.cpp`): gate * sigmoid(gate) * up fused into one kernel.
- **Fast kernels**: `fast::rope`, `fast::rms_norm`, `fast::scaled_dot_product_attention`, `quantized_matmul`.
- **Training engine** (`mlx-cpp/training/`): Muon+AdamW optimizer, gradient clipping, EMA, compiled eval, WSD/cosine LR schedules.
- **Restructured** `mlx-cpp/` into `inference/` and `training/` folders.

## Attempted but not adopted

- **ZMLX-style fused rmsnorm+residual**: manual rmsnorm (mean+square+rsqrt) replaced `fast::rms_norm`, which is already a single hand-tuned Metal kernel. Result dropped from 621 tok/s to 86 tok/s. Reverted.
- **Compiled decode with pre-allocated KV cache**: shape instability from `fast::rope` with array offsets and `split` shape inference failed. Reverted.
- **Bfloat16 forward pass**: slower due to recompilation overhead.
- **Fused full training step**: marginal gains only; Python still wins.

## Raw commands

```bash
# C++ training
cd mlx-cpp
./build/train train --quick

# C++ inference (fp32)
./build/train infer --checkpoint checkpoint.safetensors \
  --prompt "The future of AI is" --max-tokens 300 --benchmark

# C++ inference (4-bit)
./build/train infer --checkpoint checkpoint.safetensors \
  --prompt "The future of AI is" --max-tokens 300 --benchmark --quantize

# Python training
python3 benchmark_mlx_py.py

# Python inference
python3 benchmark_infer_mlx.py

# Forward-only microbenchmark
python3 benchmark_forward_py.py
./build/bench_forward
```

## Notes

- Benchmarks are sensitive to thermal state. Numbers above were taken after a cooldown period.
- The C++ forward pass is the only workload that beats Python on a like-for-like basis.
- Inference is the clear win for the C++ engine thanks to the KV cache and GPU sampling.
