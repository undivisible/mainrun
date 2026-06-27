#pragma once
#include <mlx/mlx.h>
#include <vector>

// Muon + AdamW hybrid optimizer.
// 2D non-embedding params → Muon (momentum + Newton-Schulz orthogonalization).
// 1D params + embeddings → AdamW.
// Interface: step(params, grads) → new params. State maintained internally.
class Optimizer {
public:
  Optimizer(float lr, float weight_decay, float b1, float b2, float eps);
  void set_learning_rate(float lr);
  std::vector<mlx::core::array> step(const std::vector<mlx::core::array>& params,
                                     const std::vector<mlx::core::array>& grads);

private:
  float lr_, adamw_lr_, weight_decay_, beta1_, beta2_, eps_;
  int step_ = 0;
  // Per-param buffers, indexed same as params vector.
  std::vector<mlx::core::array> muon_mom_;   // momentum buffers for muon params
  std::vector<mlx::core::array> adamw_m_;    // first moment for adamw params
  std::vector<mlx::core::array> adamw_v_;    // second moment for adamw params
  std::vector<bool> is_muon_;                // classification per param
  bool initialized_ = false;

  void init(const std::vector<mlx::core::array>& params);
  mlx::core::array newton_schulz5(const mlx::core::array& g, int steps);
};
