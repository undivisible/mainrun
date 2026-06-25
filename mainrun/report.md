# Mainrun report

## Result

| | Validation loss |
|---|----------------|
| Baseline | **1.754** |
| **Best so far (submit if nothing beats it)** | **1.233613** |

~**30%** below baseline. Rules: 7 epochs, seed 1337, same data; **`evaluate()` unchanged**.

**Best log:** `logs/run_v1_qknorm.log` (QK norm, dropout 0.1).  
**Last safe try (you run):** dropout **0.05**, same else → save as `logs/run_v1_do005.log`; submit whichever is lower.

---

## Time spent (~2–3 hours training on MPS)

| Phase | ~Wall clock |
|-------|-------------|
| v1 + failed v2 + experiment + QK + kitchen sink | **~2–3 h** train |
| Setup (venv, dataset) | extra |

---

## Submission stack (when not beating 1.233613)

**Model:** RMSNorm, RoPE, SwiGLU, SDP attention, NeoX parallel block, tied embeddings, residual init scaling, **QK norm**, **title-boundary mask** (no cross-`<eos>` attention).

**Train:** BPE 16k, AdamW, OneCycle max lr 5e-4, 128×64, accum 1 → **938** steps.

---

## All runs (including failures)

| Run | What changed | Best val | Why it failed / note |
|-----|----------------|----------|----------------------|
| **v1** | Modern block + mask + BPE | 1.234482 | Good; superseded by QK |
| **v2** | grad accum **2**, dropout 0 | **~1.284** | **Fail:** 469 optimizer steps not 938; OneCycle mismatch |
| **Experiment** | 256 ctx, unigram, ~46M, MetalMuon, WSD, R-Drop, gates, value-residual, … | **~1.448** | **Fail:** heavier than v1, worse CE under same 7 epochs |
| **QK norm** | v1 + QK norm, dropout 0.1 | **1.233613** | **Best — use if last run doesn’t beat it** |
| **Kitchen sink** | dropout **0**, title CE boost 1.2×, rope 50k, lr 5.5e-4, z-loss | **1.248** (final) | **Fail:** val improved to ~1.245 mid-run then **overfit**; train tricks ≠ better eval |
| **Last safe** | QK + **dropout 0.05** only | *pending* | One ~15 min run; if ≥ 1.2336, submit QK log |

**Not run / skipped:** accum without step retune (learned from v2), inauguration/In stack (different project), changing `evaluate()`.

**Forks (GitHub):** Most commit baseline ~1.753; one fork documents sweeps ~1.14 with week-long search—not matched here.

---

## Justifications

- **Title mask:** Titles are separate segments; cross-title attention is wrong inductive bias.
- **Modern block:** Standard small-LM upgrade vs baseline.
- **QK norm:** Measured small gain over v1.
- **Failed kitchen sink:** Train-only objectives and zero dropout did not improve **graded** CE.
- **Failed experiment:** Parameter count and optimizer complexity ≠ better under fixed epoch budget.

---

## Last run (you)

```bash
cd ~/projects/mainrun/mainrun
source .venv/bin/activate
export MAINRUN_LOCAL=1 HF_HUB_ENABLE_HF_TRANSFER=1
python train.py
cp logs/mainrun.log logs/run_v1_do005.log
```

Compare best `validation_step` to **1.233613**. Lower → submit that log + set `dropout=0.05` in `train.py`. Not lower → submit **`run_v1_qknorm.log`**, set `dropout=0.1` in `train.py` for a clean story.

**Expect:** ~1.232–1.235 or same; big jumps unlikely.

---

## Repo

Branch **`submit-v1`**. Export **`report.pdf`** before `task submit`.