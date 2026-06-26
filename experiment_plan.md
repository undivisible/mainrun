# Experiment Plan: Pushing Beyond 0.9746

## Current Best: 0.974555 (44.4% below baseline)

## Analysis of Current Run
- **Overfitting pattern**: Validation loss starts increasing around step 492, but dropout=0.15 smooths it
- **Learning dynamics**: Constant LR=0.02 throughout training
- **Model capacity**: d_model=576, 8 layers, 9 heads, ~15M parameters
- **Training efficiency**: 896 steps in ~5 hours on MacBook M2

## High-Impact Experiments to Try

### 1. Learning Rate Scheduling (Highest Priority)
**Hypothesis**: Constant LR is suboptimal; cosine decay will improve final convergence
- **Cosine with warmup**: 0.025 → 0.001 over 896 steps (100 step warmup)
- **Expected improvement**: 2-5% better final loss
- **Risk**: Low (LR scheduling is well-understood)

### 2. Architectural Variations
**Hypothesis**: Different width/depth ratios may capture patterns better

#### 2a. Wider Model (Same params)
- d_model=768, n_heads=12, n_layers=6
- **Trade-off**: Less hierarchical, more representational capacity per layer
- **Expected improvement**: 1-3%

#### 2b. Deeper Model (Same params)  
- d_model=512, n_heads=8, n_layers=10
- **Trade-off**: More hierarchical, potential vanishing gradient issues
- **Expected improvement**: 1-2%

### 3. Advanced Regularization
**Hypothesis**: Current overfitting can be reduced further

#### 3a. Adaptive Dropout
- Start at 0.05, increase to 0.2 over training
- **Rationale**: More regularization later when model starts overfitting
- **Expected improvement**: 1-3%

#### 3b. Token-level Data Augmentation
- Random token dropout (5%) and shuffling
- **Rationale**: Better generalization through noise injection
- **Expected improvement**: 1-2%

### 4. Attention Improvements
**Hypothesis**: Better attention scaling can capture longer dependencies

#### 4a. Flash Attention Variant
- Implement memory-efficient attention
- **Rationale**: Better scaling, potentially larger effective context
- **Expected improvement**: 1-2%

## Experiment Priority Order

1. **Cosine LR scheduling** (Highest ROI, lowest risk)
2. **Adaptive dropout** (Targets observed overfitting)
3. **Wider model** (Same compute, different architecture)
4. **Token augmentation** (Simple to implement)
5. **Deeper model** (Higher risk due to depth)
6. **Flash attention** (More complex implementation)

## Success Criteria
- **Minor improvement**: <0.970 (1% better)
- **Significant improvement**: <0.960 (2% better) 
- **Breakthrough**: <0.950 (3% better)

## Implementation Strategy
Given the dependency issues with the current environment, the experiments are ready to run once the proper Python environment is available. All experiment code is prepared and can be executed sequentially.

Each experiment will:
1. Run for full 896 steps
2. Track best validation loss
3. Compare against 0.974555 baseline
4. Record results in experiment_results.json

## Estimated Time
- Each experiment: ~5 hours
- Total sequential time: ~30 hours
- Can be run in parallel if multiple machines available

## Next Steps
1. Set up proper Python environment with dependencies
2. Run cosine LR experiment (highest priority)
3. If improvement found, update report and send follow-up email
4. Continue with next experiment if time permits