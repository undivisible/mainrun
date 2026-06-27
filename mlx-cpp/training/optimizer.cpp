#include "training/optimizer.h"
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
      adamw_m_.push_back(zeros_like(params[i]));
      adamw_v_.push_back(zeros_like(params[i]));
    } else {
      muon_mom_.push_back(zeros_like(params[i]));
      adamw_m_.push_back(zeros_like(params[i]));
      adamw_v_.push_back(zeros_like(params[i]));
    }
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

// Pure functional version for compile().
// state layout: [muon_mom[0..N-1], adamw_m[0..N-1], adamw_v[0..N-1]]
// Returns: [new_params[0..N-1], new_muon_mom[0..N-1], new_adamw_m[0..N-1], new_adamw_v[0..N-1]]
std::vector<array> Optimizer::step_compiled(
    const std::vector<array>& params,
    const std::vector<array>& grads,
    const std::vector<array>& state,
    const std::vector<array>& bc,
    const std::vector<array>& lr_arr) {
  size_t N = params.size();
  float momentum = 0.95f;
  float inv_bc1_f = bc[0].item<float>();
  float inv_bc2_f = bc[1].item<float>();
  float lr_f = lr_arr[0].item<float>();
  float adamw_lr_f = lr_arr[1].item<float>();
  float wd_muon_f = lr_arr[2].item<float>();
  float wd_adamw_f = lr_arr[3].item<float>();

  std::vector<array> new_params;
  std::vector<array> new_mom_vec;
  std::vector<array> new_am_vec;
  std::vector<array> new_av_vec;
  new_params.reserve(N);
  new_mom_vec.reserve(N);
  new_am_vec.reserve(N);
  new_av_vec.reserve(N);

  for (size_t i = 0; i < N; i++) {
    const auto& mom = state[i];
    const auto& am = state[N + i];
    const auto& av = state[2 * N + i];

    if (is_muon_[i]) {
      auto new_mom = mom * array(momentum) + grads[i];
      auto nesterov = grads[i] + new_mom * array(momentum);
      auto ortho = newton_schulz5(nesterov, 5);
      auto shape = params[i].shape();
      float ratio = std::max(1.0f, (float)shape[0] / (float)shape[1]);
      float scale_f = lr_f * std::sqrt(ratio);
      new_params.push_back(params[i] * array(wd_muon_f) - ortho * array(scale_f));
      new_mom_vec.push_back(new_mom);
      new_am_vec.push_back(am);
      new_av_vec.push_back(av);
    } else {
      auto new_m = am * array(beta1_) + grads[i] * array(1.0f - beta1_);
      auto new_v = av * array(beta2_) + square(grads[i]) * array(1.0f - beta2_);
      auto m_hat = new_m * array(inv_bc1_f);
      auto v_hat = new_v * array(inv_bc2_f);
      auto update = m_hat / (sqrt(v_hat) + array(eps_));
      new_params.push_back(params[i] * array(wd_adamw_f) - update * array(adamw_lr_f));
      new_mom_vec.push_back(mom);
      new_am_vec.push_back(new_m);
      new_av_vec.push_back(new_v);
    }
  }

  // Output: [new_params..., new_mom_vec..., new_am_vec..., new_av_vec...]
  std::vector<array> out;
  out.reserve(4 * N);
  for (auto& p : new_params) out.push_back(std::move(p));
  for (auto& s : new_mom_vec) out.push_back(std::move(s));
  for (auto& s : new_am_vec) out.push_back(std::move(s));
  for (auto& s : new_av_vec) out.push_back(std::move(s));
  return out;
}
