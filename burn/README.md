# Burn track (innovation appendix)

Graded metric: PyTorch `task train` + frozen `evaluate()` in `train.py`.

This crate will mirror:

- `tokenizer.json` exported from official run
- seed 1337, same title split
- val loss = sum CE / `len(val_text)` (see `mainrun/eval_metric.py`)

Backend on Mac: `burn-wgpu` (Metal). Implement after official val &lt; 1.5.

```bash
# future
task train-burn
```