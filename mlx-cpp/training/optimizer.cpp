#include "training/optimizer.h"
#include <cmath>
#include <algorithm>

using namespace mlx::core;

Optimizer::Optimizer(float lr)
    : lr_(lr), adamw_lr_(lr / 40.0f) {}

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
    muon_mom_.push_back(zeros_like(params[i]));
    adamw_m_.push_back(zeros_like(params[i]));
    adamw_v_.push_back(zeros_like(params[i]));
  }
  // Build flat state vector: [muon_mom_..., adamw_m_..., adamw_v_...]
  state_.clear();
  state_.reserve(3 * params.size());
  for (auto& m : muon_mom_) state_.push_back(m);
  for (auto& m : adamw_m_) state_.push_back(m);
  for (auto& v : adamw_v_) state_.push_back(v);

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


