#pragma once
#include <mlx/mlx.h>
#include <vector>

class Optimizer {
public:
  explicit Optimizer(float lr);
  void set_learning_rate(float lr);
  void init_state(const std::vector<mlx::core::array>& params) { init(params); }
  const std::vector<mlx::core::array>& state() const { return state_; }
  bool is_muon_param(size_t i) const { return is_muon_[i]; }
  mlx::core::array newton_schulz5(const mlx::core::array& g, int steps);

private:
  float lr_;
  float adamw_lr_;
  std::vector<mlx::core::array> muon_mom_;
  std::vector<mlx::core::array> adamw_m_;
  std::vector<mlx::core::array> adamw_v_;
  std::vector<mlx::core::array> state_;
  std::vector<bool> is_muon_;
  bool initialized_ = false;

  void init(const std::vector<mlx::core::array>& params);
};
