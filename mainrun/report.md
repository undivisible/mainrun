# Mainrun report

## Result

| | Validation loss |
|---|----------------|
| Baseline (provided) | **1.754** |
| **Submission (Run v1)** | **1.234** |

Improvement ≈ **29.6%** relative on the graded metric. Constraints respected: 7 epochs, seed 1337, fixed dataset and val split, no pretraining or augmentation, **`evaluate()` unchanged**.

Evidence: `logs/run_v1_dropout0.1_accum1.log`, best `validation_step` loss **1.234482** at step 938.

---

## Problem and metric

The task is next-token prediction on Hacker News **titles** concatenated with `<eos>`. Validation loss is total cross-entropy over the validation token stream divided by the **character length** of the joined validation text (same as baseline). Improvements must come from architecture, tokenization, optimization, and training dynamics—not from altering that evaluation.

---

## Approach (Run v1)

I treated this as a **small language model engineering** problem: adopt a modern decoder block, respect the **segment structure** of the corpus, and keep the training schedule aligned with the fixed epoch budget.

### Architecture

- **RMSNorm (pre-norm)** instead of LayerNorm — fewer parameters, stable training at depth 12.
- **RoPE** on queries and keys — relative position without a learned position embedding table.
- **SwiGLU** feed-forward — stronger FFN than GELU×4 for similar parameter use.
- **`scaled_dot_product_attention`** — identical math to manual softmax attention; uses fused kernels where available.
- **GPT-NeoX parallel block** — attention and FFN branches both read the same normalized residual stream, then add back; often optimizes better than strict serial pre-norm.
- **Tied embeddings** — `lm_head` shares weights with token embeddings (~27M trainable params total).
- **Residual projection init** — attention and MLP output linear layers initialized with std `0.02 / √(2 × n_layer)` so depth does not blow up activation variance at step zero.

**Shape:** 12 layers, 12 heads, **d_model 384**, context **128**.

### Title-boundary attention mask

Training text is `title₁ <eos> title₂ <eos> …`. Pure causal masking still lets tokens in one headline attend to another headline in the same window. I mask to **causal ∧ same segment**, where segment boundaries come from `<eos>` positions. This matches the generative structure of the data. Tokenization, batching, and `evaluate()` are unchanged; only allowed attention pairs in `forward` differ.

### Tokenization and optimization

- Byte-level **BPE**, vocabulary **16 000**, specials `<pad>`, `<eos>`, `<unk>`.
- Corpus pretokenized once to flat tensors (`data.py`) for simple batching.
- **AdamW** with weight decay on **2D** parameters only; **OneCycleLR** with max lr **5×10⁻⁴**.
- **Batch 64**, **dropout 0.1**, **gradient accumulation 1** → **938 optimizer steps** over seven epochs (134 micro-batches per epoch).

Development was done on Apple Silicon (MPS); submission path remains `task train` in the Dev Container per README.

---

## What we tried and rejected

| Attempt | Outcome | Lesson |
|---------|---------|--------|
| **Grad accum = 2**, dropout 0 | Val **~1.28** vs v1 **1.234** | Fixed 7 epochs ⇒ accum halves **optimizer steps** (469 vs 938); OneCycle was tuned for 938—schedule and update count must move together. |
| **Larger “sweep-style” stack** (256 ctx, unigram, ~46M params, Muon, R-Drop, many extras) | Val **~1.45** after full 7 epochs | More machinery and public “winning” recipes did **not** beat the simpler v1 recipe on this metric and budget. |
| **Optional v3-style QK norm** on v1 shape | Off by default (`qk_norm=False`) | Implemented in `model.py` for a cheap ablation; **submit config matches the 1.234 log** without it. |

Public GitHub forks mostly commit only **baseline.log (~1.753)**. Published sub-1.2 numbers elsewhere rely on large sweep campaigns, not a single default `train.py` run.

---

## Repository map

| File | Role |
|------|------|
| `train.py` | Entry point; hyperparameters; training loop; frozen `evaluate()` |
| `model.py` | GPT, blocks, RoPE, title mask |
| `rope.py` | Rotary embeddings |
| `data.py` | Dataset load, BPE training, batches |
| `utils.py` | Devcontainer guard; `MAINRUN_LOCAL` for local runs |

Refactor is intentional; behavior is still `python train.py` / `task train`.

---

## Reproducibility

```bash
git checkout submit-v1
cd mainrun
export MAINRUN_LOCAL=1   # local only
python train.py
```

Optional ablation on same branch: set `qk_norm=True` and/or `dropout=0.05` in `Hyperparameters`, re-run, compare `validation_step` to **1.234** before changing the submission log.

---

## Conclusion

Run v1 combines a **standard modern LM block** with one **dataset-specific** idea (title-boundary masking) and a **schedule matched to the step budget**. It delivers a large, reproducible gain over baseline without touching the evaluation function. Heavier configurations inspired by public sweeps were tried and did not improve the graded loss under the same rules.

Export this document to **`report.pdf`** in this folder before `task submit`.