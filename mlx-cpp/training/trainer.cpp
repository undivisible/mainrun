#include "training/trainer.h"
#include "model.h"
#include "training/optimizer.h"
#include "inference/engine.h"
#include <mlx/transforms.h>
#include <mlx/compile.h>
#include <mlx/memory.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <utility>

using namespace mlx::core;

// log_softmax along last axis: x - logsumexp(x, -1, keepdims)
static array log_softmax_last(const array& logits) {
  return logits - logsumexp(logits, -1, true);
}

// cross-entropy with sum reduction (for evaluation): total = sum over all tokens
static array cross_entropy_sum(const array& logits, const array& targets) {
  auto lp = log_softmax_last(logits);
  auto picked = take_along_axis(lp, expand_dims(targets, -1), -1);
  return -sum(picked);
}

float wsd_lr(int step, int max_steps, float max_lr, float warmup_pct, float decay_pct) {
  int warmup = std::max(1, (int)(max_steps * warmup_pct));
  int decay_start = max_steps - std::max(1, (int)(max_steps * decay_pct));
  int decay_len = std::max(1, max_steps - decay_start);
  if (step < warmup) return max_lr * (float)(step + 1) / warmup;
  if (step < decay_start) return max_lr;
  return max_lr * std::max(0.0f, 1.0f - (float)(step - decay_start + 1) / decay_len);
}

// EMA shadow params with warmup decay: min(target, (1+step)/(10+step))
struct ModelEMA {
  std::vector<array> shadow;
  std::vector<array> backup;
  float target;
  explicit ModelEMA(const std::vector<array>& params, float target_decay) : target(target_decay) {
    for (const auto& p : params) shadow.push_back(p);
  }
  static float decay(int step, float t) { return std::min(t, (float)(step + 1) / (10 + step)); }
  void update(const std::vector<array>& params, int step) {
    float d = decay(step, target);
    array da(d), one_minus(1.0f - d);
    for (size_t i = 0; i < shadow.size(); i++)
      shadow[i] = shadow[i] * da + params[i] * one_minus;
  }
  std::vector<array> swap_in(const std::vector<array>& params) {
    backup = {};
    for (const auto& p : params) backup.push_back(p);
    return shadow;
  }
  std::vector<array> swap_out() { return backup; }
};

// Compiled eval: forward + cross_entropy for a single batch.
// No randomness (no dropout), so the graph is stable and compilation works.
static std::vector<array> eval_single_batch_impl(const std::vector<array>& inputs) {
  extern GPT* g_eval_model;
  extern int g_eval_B, g_eval_T, g_eval_V;
  size_t np = inputs.size() - 2;
  std::vector<array> p(inputs.begin(), inputs.begin() + np);
  g_eval_model->set_parameters(p);
  auto logits = g_eval_model->forward(inputs[np], false);
  auto lf = reshape(logits, {g_eval_B * g_eval_T, g_eval_V});
  auto tf = reshape(inputs[np + 1], {g_eval_B * g_eval_T});
  return {cross_entropy_sum(lf, tf)};
}

GPT* g_eval_model = nullptr;
int g_eval_B = 0, g_eval_T = 0, g_eval_V = 0;

static float eval_loss_compiled(GPT& model, DataLoader& data,
                                const std::function<std::vector<array>(const std::vector<array>&)>& compiled_eval,
                                const std::vector<array>& params) {
  data.reset_val_ptr();
  int B = data.batch_size(), T = data.seq_len(), V = data.vocab_size();
  g_eval_model = &model;
  g_eval_B = B; g_eval_T = T; g_eval_V = V;
  array total = array(0.0f);
  while (auto b = data.next_val_batch()) {
    auto& [x, y] = *b;
    std::vector<array> inputs;
    inputs.reserve(params.size() + 2);
    for (const auto& p : params) inputs.push_back(p);
    inputs.push_back(x);
    inputs.push_back(y);
    auto result = compiled_eval(inputs);
    total = total + result[0];
  }
  eval(total);
  return total.item<float>() / (float)data.val_char_count();
}

float run_training(DataLoader& data, const TrainConfig& cfg) {
  GPTConfig mc;
  mc.vocab_size = data.vocab_size();
  mc.block_size = cfg.block_size;
  mc.n_layer = cfg.n_layer;
  mc.n_head = cfg.n_head;
  mc.d_model = cfg.d_model;
  mc.dropout = cfg.dropout;
  mc.eos_id = data.eos_id();
  mc.rope_theta = cfg.rope_theta;
  mc.qk_norm = cfg.qk_norm;
  GPT model(mc);

  std::vector<array> params;
  for (auto* p : model.parameters()) params.push_back(*p);
  fprintf(stderr, "param tensors: %zu\n", params.size());

  Optimizer opt(cfg.max_lr);
  ModelEMA ema(params, cfg.ema_target_decay);

  std::vector<int> argnums(params.size());
  for (size_t i = 0; i < params.size(); i++) argnums[i] = (int)i;

  float best_val = 1e9f;
  auto t_start = std::chrono::high_resolution_clock::now();

  size_t N = params.size();

  // --- Approach: compile the entire training step (vg + optimizer) in one graph ---
  // Pure functional loss: parameters are explicit inputs, no model state mutation.
  // This makes the graph transparent to value_and_grad/compile.
  auto loss_fn = std::function<array(const std::vector<array>&)>(
    [&](const std::vector<array>& inputs) {
      size_t np = inputs.size() - 2;
      std::vector<array> p(inputs.begin(), inputs.begin() + np);
      return model.loss_functional(p, inputs[np], inputs[np + 1], true);
    });
  auto vg = value_and_grad(loss_fn, argnums);

  // Initialize optimizer state
  opt.init_state(params);
  auto& state = const_cast<std::vector<array>&>(opt.state());

  // Fused step: input [params(N), state(3N), idx, targets, bc1, bc2, lr, adamw_lr, wd_muon, wd_adamw]
  // Output: [loss, new_params(N), new_state(3N)]
  auto fused_step = std::function<std::vector<array>(const std::vector<array>&)>(
    [&, vg, N](const std::vector<array>& inputs) -> std::vector<array> {
      std::vector<array> p(inputs.begin(), inputs.begin() + N);
      std::vector<array> s(inputs.begin() + N, inputs.begin() + 4 * N);
      const array& idx = inputs[4 * N];
      const array& targets = inputs[4 * N + 1];
      const array& inv_bc1 = inputs[4 * N + 2];
      const array& inv_bc2 = inputs[4 * N + 3];
      const array& lr_a = inputs[4 * N + 4];
      const array& adamw_lr_a = inputs[4 * N + 5];
      const array& wd_muon_a = inputs[4 * N + 6];
      const array& wd_adamw_a = inputs[4 * N + 7];

      // Forward + backward
      std::vector<array> vg_inputs;
      vg_inputs.reserve(N + 2);
      for (auto& pp : p) vg_inputs.push_back(pp);
      vg_inputs.push_back(idx);
      vg_inputs.push_back(targets);
      auto [loss, grads] = vg(vg_inputs);

      // Clip grad norm
      array total_sq = array(0.0f);
      for (const auto& g : grads) total_sq = total_sq + sum(square(g));
      array clip_scale = minimum(array(1.0f), array(1.0f) / sqrt(total_sq));
      for (auto& g : grads) g = g * clip_scale;

      // Optimizer inline
      array momentum_a(0.95f), beta1_a(0.9f), beta2_a(0.95f), eps_a(1e-8f);
      std::vector<array> new_params, new_mom, new_am, new_av;
      new_params.reserve(N); new_mom.reserve(N); new_am.reserve(N); new_av.reserve(N);
      for (size_t i = 0; i < N; i++) {
        if (opt.is_muon_param(i)) {
          auto mom_update = s[i] * momentum_a + grads[i];
          auto nesterov = grads[i] + mom_update * momentum_a;
          // Three Newton-Schulz iterations balance optimizer quality and graph cost.
          auto ortho = opt.newton_schulz5(nesterov, 3);
          auto shape = p[i].shape();
          float ratio = std::max(1.0f, (float)shape[0] / (float)shape[1]);
          auto scale = lr_a * array(std::sqrt(ratio));
          new_params.push_back(p[i] * wd_muon_a - ortho * scale);
          new_mom.push_back(std::move(mom_update));
          new_am.push_back(s[N + i]);
          new_av.push_back(s[2 * N + i]);
        } else {
          auto new_m = s[N + i] * beta1_a + grads[i] * array(0.1f);
          auto new_v = s[2 * N + i] * beta2_a + square(grads[i]) * array(0.05f);
          auto update = (new_m * inv_bc1) / (sqrt(new_v * inv_bc2) + eps_a);
          new_params.push_back(p[i] * wd_adamw_a - update * adamw_lr_a);
          new_mom.push_back(s[i]);
          new_am.push_back(std::move(new_m));
          new_av.push_back(std::move(new_v));
        }
      }
      std::vector<array> out;
      out.reserve(1 + 4 * N);
      out.push_back(loss);
      for (auto& np : new_params) out.push_back(std::move(np));
      for (auto& m : new_mom) out.push_back(std::move(m));
      for (auto& m : new_am) out.push_back(std::move(m));
      for (auto& m : new_av) out.push_back(std::move(m));
      return out;
    });
  auto compiled_step = compile(fused_step);  // shapeless default

  // Compiled eval function (no dropout → stable graph)
  auto compiled_eval = compile(std::function<std::vector<array>(const std::vector<array>&)>(eval_single_batch_impl));

  // Accumulate loss on GPU to avoid per-step CPU sync
  array loss_accum = array(0.0f);
  int loss_count = 0;

  for (int step = 0; step < cfg.max_steps; step++) {
    float lr = wsd_lr(step, cfg.max_steps, cfg.max_lr, cfg.warmup_pct, cfg.decay_pct);
    opt.set_learning_rate(lr);

    auto [x, y] = data.get_train_batch();

    // --- Single compiled step (forward + backward + clip + optimizer) ---
    int s = step + 1;
    float bc1_f = 1.0f - std::pow(0.9f, s);
    float bc2_f = 1.0f - std::pow(0.95f, s);
    float adamw_lr_f = lr * 0.2f;
    float wd_muon_f = 1.0f - lr * 0.1f;
    float wd_adamw_f = 1.0f - adamw_lr_f * 0.1f;

    std::vector<array> step_inputs;
    step_inputs.reserve(4 * N + 8);
    for (auto& p : params) step_inputs.push_back(p);
    for (auto& st : state) step_inputs.push_back(st);
    step_inputs.push_back(x);
    step_inputs.push_back(y);
    step_inputs.push_back(array(1.0f / bc1_f));
    step_inputs.push_back(array(1.0f / bc2_f));
    step_inputs.push_back(array(lr));
    step_inputs.push_back(array(adamw_lr_f));
    step_inputs.push_back(array(wd_muon_f));
    step_inputs.push_back(array(wd_adamw_f));

    auto step_result = compiled_step(step_inputs);

    // Accumulate loss on GPU (lazy)
    loss_accum = loss_accum + step_result[0];
    loss_count++;

    // Update params + state
    params.assign(step_result.begin() + 1, step_result.begin() + 1 + N);
    state.assign(step_result.begin() + 1 + N, step_result.end());
    model.set_parameters(params);
    if (cfg.use_ema) ema.update(params, step);

    if (step == 0 || step % cfg.eval_interval == 0 || step == cfg.max_steps - 1) {
      // Sync loss for logging
      eval(loss_accum);
      float avg_loss = loss_accum.item<float>() / loss_count;
      loss_accum = array(0.0f);
      loss_count = 0;

      float val_ema = 0.0f;
      if (cfg.use_ema) {
        auto ema_params = ema.swap_in(params);
        val_ema = eval_loss_compiled(model, data, compiled_eval, ema_params);
        model.set_parameters(ema.swap_out());
      } else {
        val_ema = eval_loss_compiled(model, data, compiled_eval, params);
      }
      auto now = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(now - t_start).count();
      if (val_ema < best_val) best_val = val_ema;
      printf("Step %d: train=%.4f, val_ema=%.4f, lr=%.5f, %.1fs elapsed, %.0fms/step\n",
             step, avg_loss, val_ema, lr, elapsed, elapsed * 1000 / (step + 1));
      fflush(stdout);
    }
  }
  save_checkpoint("checkpoint.safetensors", model);
  return best_val;
}
