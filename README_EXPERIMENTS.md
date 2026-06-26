# Experiments: Pushing Beyond 0.9746

## Quick Start

```bash
# Run the highest-impact experiment (cosine LR scheduling)
./run_experiments.sh quick

# Run all experiments sequentially (takes ~25 hours)
./run_experiments.sh all

# Run a specific experiment
./run_experiments.sh cosine_lr
./run_experiments.sh wider_model
```

## Current Status
- **Best result**: 0.974555 (44.4% below baseline)
- **Next target**: <0.970 (1% improvement)
- **Environment**: Ready to run, needs proper Python dependencies

## Experiment Details

### 1. Cosine LR Scheduling (Highest Priority)
- **What**: Replace constant LR=0.02 with cosine decay (0.025→0.001)
- **Why**: Better final convergence, less overfitting
- **Expected**: 2-5% improvement
- **Runtime**: ~5 hours

### 2. Wider Model
- **What**: d_model=768, 12 heads, 6 layers (same params)
- **Why**: More representational capacity per layer
- **Expected**: 1-3% improvement
- **Runtime**: ~5 hours

### 3. Deeper Model
- **What**: d_model=512, 8 heads, 10 layers (same params)
- **Why**: More hierarchical feature extraction
- **Expected**: 1-2% improvement
- **Runtime**: ~5 hours

### 4. Adaptive Dropout
- **What**: Start at 0.05, increase to 0.2 during training
- **Why**: More regularization when overfitting starts
- **Expected**: 1-3% improvement
- **Runtime**: ~5 hours

### 5. Token Augmentation
- **What**: Random token dropout (5%) and shuffling
- **Why**: Better generalization through noise
- **Expected**: 1-2% improvement
- **Runtime**: ~5 hours

## Results Tracking

All results are saved to `experiment_results.json`:

```json
[
  {
    "name": "cosine_lr",
    "description": "Cosine learning rate with warmup",
    "best_loss": 0.968,
    "duration_seconds": 18000,
    "timestamp": "2026-06-26T10:30:00Z",
    "status": "completed"
  }
]
```

## Success Criteria

- **Minor improvement**: <0.970 (beat current by 1%)
- **Significant improvement**: <0.960 (beat current by 2%)
- **Breakthrough**: <0.950 (beat current by 3%)

## If You Find Improvement

1. **Stop other experiments** - you have a new best!
2. **Update the report**: Add new findings to `mainrun/report.md`
3. **Send follow-up email**: "Quick update - managed to improve further to X.XXX"
4. **Submit new version**: Use the updated submission scripts

## Files Created

- `experiments.py` - Full experiment framework
- `cosine_experiment.py` - Quick cosine LR test
- `run_experiments.sh` - Bash runner script
- `experiment_plan.md` - Detailed analysis and plan
- `train_lr_schedule.py` - Enhanced training with LR scheduling

## Next Steps

1. Run `./run_experiments.sh quick` for the highest-impact experiment
2. If it improves, stop and update your submission
3. If not, try the next experiment in priority order

The experiments are designed to be conservative - each targets a specific weakness observed in the current run while keeping the overall approach that got you to 0.9746.