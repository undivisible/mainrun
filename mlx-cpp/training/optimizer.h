#pragma once
#include <mlx/mlx.h>
#include <vector>

class Optimizer {
public:
  Optimizer(float lr, float weight_decay, float b1, float b2, float eps);
  void set_learning_rate(float lr);
  void set_step(int s) { step_ = s; }
  std::vector<mlx::core::array> step(const std::vector<mlx::core::array>& params,
                                     const std::vector<mlx::core::array>& grads);

  // Compiled step: all state passed explicitly as inputs/outputs.
  // state = [muon_mom_..., adamw_m_..., adamw_v_...]
  // bc = [bc1, bc2] (bias correction factors, passed as input for compilability)
  // Returns: [new_params..., new_state...]
  std::vector<mlx::core::array> step_compiled(
      const std::vector<mlx::core::array>& params,
      const std::vector<mlx::core::array>& grads,
      const std::vector<mlx::core::array>& state,
      const std::vector<mlx::core::array>& bc,
      const std::vector<mlx::core::array>& lr_arr);

  const std::vector<mlx::core::array>& state() const { return state_; }
  bool initialized() const { return initialized_; }
  void init_state(const std::vector<mlx::core::array>& params) { init(params); }
  bool is_muon_param(size_t i) const { return is_muon_[i]; }
  mlx::core::array newton_schulz5(const mlx::core::array& g, int steps);

private:
  float lr_, adamw_lr_, weight_decay_, beta1_, beta2_, eps_;
  int step_ = 0;
  std::vector<mlx::core::array> muon_mom_;
  std::vector<mlx::core::array> adamw_m_;
  std::vector<mlx::core::array> adamw_v_;
  std::vector<mlx::core::array> state_;
  std::vector<bool> is_muon_;
  bool initialized_ = false;

  void init(const std::vector<mlx::core::array>& params);
};
