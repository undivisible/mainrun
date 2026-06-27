#include "optimizer.h"
#include <cmath>
#include <algorithm>

using namespace mlx::core;

Optimizer::Optimizer(float lr, float weight_decay, float b1, float b2, float eps)
    : lr_(lr), adamw_lr_(lr / 40.0f), weight_decay_(weight_decay),
      beta1_(b1), beta2_(b2), eps_(eps) {}

void Optimizer::set_learning_rate(float lr) {
  lr_ = lr;
  adamw_lr_ = lr / 40.0f;
}

// Classify params: 2D and not the first param (token_emb, which is embedding) → Muon.
// All 1D params and the embedding → AdamW.
// ponytail: index 0 is always token_emb in our model. Simpler than name matching.
void Optimizer::init(const std::vector<array>& params) {
  is_muon_.resize(params.size(), false);
  muon_mom_.clear();
  muon_mom_.reserve(params.size());
  adamw_m_.clear();
  adamw_m_.reserve(params.size());
  adamw_v_.clear();
  adamw_v_.reserve(params.size());
  for (size_t i = 0; i < params.size(); i++) {
    bool is_2d = params[i].ndim() == 2;
    bool is_embedding = (i == 0);  // token_emb is first param, 2D but AdamW
    is_muon_[i] = is_2d && !is_embedding;
    if (is_muon_[i]) {
      muon_mom_.push_back(zeros_like(params[i]));
      adamw_m_.push_back(array(0.0f));
      adamw_v_.push_back(array(0.0f));
    } else {
      muon_mom_.push_back(array(0.0f));  // placeholder, never used
      adamw_m_.push_back(zeros_like(params[i]));
      adamw_v_.push_back(zeros_like(params[i]));
    }
  }
  initialized_ = true;
}

// Newton-Schulz quintic orthogonalization (5 steps).
// Matches PyTorch _muon.py exactly.
array Optimizer::newton_schulz5(const array& g, int steps) {
  auto shape = g.shape();
  int rows = shape[0], cols = shape[1];
  // Transpose so rows <= cols (wide matrix). Ensures Gram X@X.T is small and full-rank.
  array x = (rows > cols) ? transpose(g, {1, 0}) : g;

  // Frobenius normalize — stay on GPU, avoid CPU sync
  auto frob = sqrt(sum(square(x)));
  // ponytail: clamp on GPU to avoid div-by-zero, no CPU roundtrip
  x = x / maximum(frob, array(1e-8f));

  const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
  for (int s = 0; s < steps; s++) {
    auto xt = transpose(x, {1, 0});
    auto gram = matmul(x, xt);
    auto gram_sq = matmul(gram, gram);
    auto gram_update = gram * array(b) + gram_sq * array(c);
    // PyTorch: ortho_grad = a * ortho_grad + gram_update @ ortho_grad
    x = x * array(a) + matmul(gram_update, x);
  }

  return (rows > cols) ? transpose(x, {1, 0}) : x;
}

std::vector<array> Optimizer::step(const std::vector<array>& params,
                                   const std::vector<array>& grads) {
  if (!initialized_) init(params);
  step_++;

  std::vector<array> out;
  out.reserve(params.size());

  // Bias correction for AdamW
  float bc1 = 1.0f - std::pow(beta1_, step_);
  float bc2 = 1.0f - std::pow(beta2_, step_);

  for (size_t i = 0; i < params.size(); i++) {
    if (is_muon_[i]) {
      // --- Muon update (matches PyTorch torch.optim.Muon) ---
      float momentum = 0.95f;

      // buf = momentum * buf + grad  (NO 1-momentum scaling, unlike standard SGD momentum)
      muon_mom_[i] = muon_mom_[i] * array(momentum) + grads[i];

      // Nesterov: update = grad + momentum * buf
      auto nesterov = grads[i] + muon_mom_[i] * array(momentum);

      // Orthogonalize
      auto ortho = newton_schulz5(nesterov, 5);

      // LR adjustment: sqrt(max(1, A/B)) where A=rows, B=cols
      auto shape = params[i].shape();
      float ratio = std::max(1.0f, (float)shape[0] / (float)shape[1]);
      float scale = lr_ * std::sqrt(ratio);

      // Decoupled weight decay + update
      float wd_scale = 1.0f - lr_ * weight_decay_;
      out.push_back(params[i] * array(wd_scale) - ortho * array(scale));
    } else {
      // --- AdamW update ---
      adamw_m_[i] = adamw_m_[i] * array(beta1_) + grads[i] * array(1.0f - beta1_);
      adamw_v_[i] = adamw_v_[i] * array(beta2_) + square(grads[i]) * array(1.0f - beta2_);

      auto m_hat = adamw_m_[i] * array(1.0f / bc1);
      auto v_hat = adamw_v_[i] * array(1.0f / bc2);
      auto update = m_hat / (sqrt(v_hat) + array(eps_));

      float wd_scale = 1.0f - adamw_lr_ * weight_decay_;
      out.push_back(params[i] * array(wd_scale) - update * array(adamw_lr_));
    }
  }

  return out;
}
