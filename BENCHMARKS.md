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
| Training (Muon+AdamW, 3 NS iters) | 694 ms/step | 453 ms/step | C++ 1.5x faster |
| Forward only (B=32, T=256) | 66 ms | 61 ms | C++ ~8% faster |
| Inference decode (KV cache, fp32) | 365 tok/s* | 1270 tok/s | C++ 3.5x faster |
| Inference decode (KV cache, 4-bit) | — | 1900 tok/s | C++ only |

*Python inference benchmark is full forward with no KV cache.

## Training

We measured the full training step with equivalent logic in both engines:

- Same model architecture
- Same Muon+AdamW optimizer (Newton-Schulz 3 steps for Muon)
- Same weight decay and bias correction
- `mx.compile()` in Python, `mlx::core::compile()` in C++

**Results (single clean run, after cooldown):**
- **Python MLX:** 694 ms/step
- **C++ MLX:** 453 ms/step

Both use the same Muon+AdamW optimizer with 3 Newton-Schulz iterations and `mx.compile`/`mlx::core::compile`. The C++ engine wins because the fused training step compiles forward + backward + clip + optimizer into a single graph, and the optimizer update is inlined as array operations rather than running through Python loops.

## Inference

| Variant | Prefill | Decode speed |
|---------|---------|--------------|
| Python MLX (full forward, no KV cache) | 4.2 ms (256 tokens) | 365 tok/s |
| C++ MLX (KV cache, fp32) | 7.5 ms | 1270 tok/s |
| C++ MLX (KV cache, 4-bit) | 4.5 ms | 1900 tok/s |

The C++ inference engine is faster because it:

1. **Uses a KV cache** — avoids recomputing past tokens each step.
2. **Uses fast MLX primitives** — `fast::rms_norm`, `fast::rope`, `fast::scaled_dot_product_attention`.
3. **Moves sampling to the GPU** — `argmax` and `categorical` stay on Metal, avoiding CPU sync.
4. **Eliminates per-token CPU-GPU sync** — the decode loop keeps argmax results on GPU and feeds them directly to the next step, only syncing every 128 steps to limit graph depth. This removes the CPU-GPU round-trip bottleneck that limited throughput to ~1273 tok/s, achieving ~1900 tok/s instead.
5. **Supports 4-bit weight quantization** — `quantized_matmul` reduces memory bandwidth and matmul cost.

## Optimizations implemented

- **Inference engine** (`mlx-cpp/inference/`): checkpoint load, tokenizer via Python helper, KV cache, GPU sampling, optional 4-bit quantization.
- **Compiled SwiGLU** (`mlx-cpp/model.cpp`): gate * sigmoid(gate) * up fused into one kernel.
- **Fast kernels**: `fast::rope`, `fast::rms_norm`, `fast::scaled_dot_product_attention`, `quantized_matmul`.
- **Avoid CPU sync in decode loop**: `forward_decode_growing` takes `int prev_len` instead of an array, removing a per-step GPU→CPU sync.
- **GPU-only decode loop**: benchmark keeps argmax on GPU across steps, feeds directly to next forward pass, syncs only every 128 steps. Eliminates CPU-GPU round-trip per token.
- **Training engine** (`mlx-cpp/training/`): Muon+AdamW optimizer, gradient clipping, EMA, compiled eval, WSD/cosine LR schedules.
- **Restructured** `mlx-cpp/` into `inference/` and `training/` folders.

## Attempted but not adopted

- **ZMLX-style fused rmsnorm+residual**: manual rmsnorm (mean+square+rsqrt) replaced `fast::rms_norm`, which is already a single hand-tuned Metal kernel. Result dropped from 621 tok/s to 86 tok/s. Reverted.
- **Compiled decode with pre-allocated KV cache**: shape instability from `fast::rope` with array offsets and `split` shape inference failed. Reverted.
- **Compiled growing-cache decode**: `split` and `slice` cannot infer output shapes with `shapeless=true` compile. Reverted.
- **Stable cache for quantized**: scatter+slice path avoids O(N²) concatenate but was slower at 200 tokens (1150 vs 1273 tok/s) due to scatter overhead. Compiled mask-based path was also slower (1108 tok/s) due to attending over full block_size. Reverted.
- **Float16 inference**: `float16` weights/activations and cache were slower than `bfloat16` (likely due to recompilation overhead). Reverted.
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
- C++ training is faster than Python MLX on the same model and optimizer.
- C++ inference is faster than Python thanks to the KV cache, GPU sampling, and 4-bit quantization.
