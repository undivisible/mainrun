# Speed Benchmarks: C++ MLX Engine vs Python MLX

**Hardware:** MacBook Pro 14-inch (M5 Pro, 2026) — 5+10 core CPU @ 4.61 GHz, 16-core GPU @ 1.62 GHz, 48 GB unified memory

**Methodology:** All numbers are the median of 5 runs taken after a full cooldown, with the first run (compile warmup) discarded. Both engines use identical model weights, architecture, and hyperparameters. Numbers were stable across runs (±2%).

Model:

- 8 layers, 9 heads, d_model 576, head_dim 64
- SwiGLU MLP (hidden ~1536), 24k vocab
- 45.7M parameters, 74 parameter tensors
- Training: batch 32, sequence 256
- Inference: greedy decode, prompt "The future of AI is", 200 tokens

## Summary

| Workload | Python MLX | C++ MLX | Notes |
|----------|------------|---------|-------|
| Training (Muon+AdamW, 3 NS iters) | 694 ms/step | 436 ms/step | 1.6x, apples-to-apples |
| Forward only (B=32, T=256) | 66 ms | 61 ms | ~8%, near parity |
| Inference decode (fp32, KV cache) | — | 1270 tok/s | different architecture, see below |
| Inference decode (4-bit, KV cache) | — | 1900 tok/s | changes weight precision |

## Training

The cleanest comparison in the set. Both engines implement the same thing:

- Same model architecture, same compiled forward pass
- Same Muon+AdamW optimizer: 3 Newton-Schulz iterations, momentum, bias correction
- Same gradient clipping and EMA
- `mx.compile()` in Python, `mlx::core::compile()` in C++

**Results (median of 5 runs, warmup discarded):**
- **Python MLX:** 694 ms/step
- **C++ MLX:** 436 ms/step — **1.6x faster**

The win comes from inlining the optimizer into the compiled graph. In Python, each optimizer step runs through Python loops over 74 parameter tensors even inside `mx.compile`, adding interpreter overhead and graph fragmentation. In C++, forward, backward, gradient clipping, and all optimizer updates are expressed as a single array-operation graph and compiled together. The Newton-Schulz iterations (3 matrix multiplications per Muon parameter) are where the overhead difference compounds most.

## Forward-only

61 ms vs 66 ms is near parity. This is expected: when there's no Python loop overhead (just a single compiled forward pass), both engines hit the same MLX/Metal kernels. The small gap is C++ avoiding some Python object dispatch. Not a major claim.

## Inference

**The comparison to the Python inference script (365 tok/s) is intentionally not framed as a direct speed ratio**, because the two implementations are architecturally different:

| Feature | Python script | C++ engine |
|---------|--------------|------------|
| KV cache | No — full forward every step | Yes |
| Sampling | CPU (`item()` + Python) | GPU-resident argmax/categorical |
| Decode loop sync | Per-token CPU-GPU round-trip | Every 128 steps |
| Weight quantization | No | Optional 4-bit |

The real claim is: **the C++ inference engine is a better-engineered decode loop**, not just a faster language. The 365 tok/s Python number is included as a baseline reference, not a fair competitor.

**Results (median of 5 runs, 200 tokens, greedy):**
- **C++ fp32:** 1270 tok/s
- **C++ 4-bit:** 1900 tok/s

The jump from fp32 to 4-bit (1.5x) comes from reduced memory bandwidth on weight loads — the model is memory-bandwidth-bound at T=1 decode, so halving weight size has a direct effect. Note that 4-bit quantization changes the weight precision and may affect output quality.

The key engineering win in the decode loop: keeping argmax GPU-resident and feeding it directly into the next forward pass eliminates the CPU-GPU round-trip that was the bottleneck. Syncing every 128 steps instead of every token limits Metal graph depth without stalling the pipeline. This alone took fp32 from ~995 tok/s to 1270, and 4-bit from ~1273 to 1900.

## Optimizations implemented

- **Fused training graph**: forward + backward + gradient clip + Muon + AdamW compiled into one `mlx::core::compile` call.
- **Compiled SwiGLU**: gate × sigmoid(gate) × up fused into one kernel, for both training and inference.
- **Fast MLX primitives**: `fast::rms_norm`, `fast::rope`, `fast::scaled_dot_product_attention`, `quantized_matmul`.
- **GPU-resident decode loop**: argmax stays on GPU, feeds directly to next step, syncs every 128 steps to limit graph depth.
- **4-bit weight quantization**: `quantized_matmul` on all linear layers during inference.
- **KV cache**: avoids recomputing past-token attention at every decode step.

## Attempted but not adopted

- **Custom Metal kernels (fused residual add, attention decode)**: both caused regressions. `fast::scaled_dot_product_attention` already dispatches to an optimized T=1 Metal path; replacing it manually was slower.
- **Compiled growing-cache decode**: `split` and `slice` can't infer output shapes under `shapeless=true`. Abandoned.
- **Stable KV cache with scatter**: atomic scatter overhead outweighed the O(N) concatenate savings (1150 vs 1273 tok/s). Reverted.
- **slice_update instead of concatenate**: slower at both 200 tokens (1600 vs 1900) and 500 tokens (535 vs 555). MLX's concatenate is already well-optimised for this pattern. Reverted.
- **Float16 activations**: slower than bfloat16, likely due to recompilation. Reverted.
- **Fused full training step (single compile across steps)**: no meaningful gain once the per-step graph is already compiled.

## Raw commands

```bash
# C++ training
cd mlx-cpp
./build/train train --quick

# C++ inference (fp32)
./build/train infer --checkpoint checkpoint.safetensors \
  --prompt "The future of AI is" --max-tokens 200 --benchmark

# C++ inference (4-bit)
./build/train infer --checkpoint checkpoint.safetensors \
  --prompt "The future of AI is" --max-tokens 200 --benchmark --quantize

# Python training
python3 benchmarks/benchmark_mlx_py.py

# Python inference baseline
python3 benchmarks/benchmark_infer_mlx.py

# Forward-only microbenchmark
python3 benchmarks/benchmark_forward_py.py
./build/bench_forward
```
