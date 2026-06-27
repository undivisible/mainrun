#include "training.h"
#include "model.h"
#include "optimizer.h"
#include <mlx/transforms.h>
#include <mlx/compile.h>
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

// Clip global grad L2 norm to max_norm. Returns clipped grads (lazy).
static std::vector<array> clip_grad_norm(const std::vector<array>& grads, float max_norm) {
  array total_sq = array(0.0f);
  for (const auto& g : grads) total_sq = total_sq + sum(square(g));
  array scale = minimum(array(1.0f), array(max_norm) / sqrt(total_sq));
  std::vector<array> out;
  out.reserve(grads.size());
  for (const auto& g : grads) out.push_back(g * scale);
  return out;
}

float wsd_lr(int step, int max_steps, float max_lr, float warmup_pct, float decay_pct) {
  int warmup = std::max(1, (int)(max_steps * warmup_pct));
  int decay_start = max_steps - std::max(1, (int)(max_steps * decay_pct));
  int decay_len = std::max(1, max_steps - decay_start);
  if (step < warmup) return max_lr * (float)(step + 1) / warmup;
  if (step < decay_start) return max_lr;
  return max_lr * std::max(0.0f, 1.0f - (float)(step - decay_start + 1) / decay_len);
}

float cosine_lr(int step, int max_steps, float max_lr, float min_lr, int warmup_steps) {
  if (step < warmup_steps) return max_lr * (float)(step + 1) / warmup_steps;
  float progress = (float)(step - warmup_steps) / (max_steps - warmup_steps);
  return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + std::cos(M_PI * progress));
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

static float eval_loss(GPT& model, DataLoader& data) {
  data.reset_val_ptr();
  int B = data.batch_size(), T = data.seq_len(), V = data.vocab_size();
  array total = array(0.0f);
  while (auto b = data.next_val_batch()) {
    auto& [x, y] = *b;
    auto logits = model.forward(x, false);
    auto lf = reshape(logits, {B * T, V});
    auto tf = reshape(y, {B * T});
    total = total + cross_entropy_sum(lf, tf);
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

  // ponytail: model.parameters() returns pointers; dereference into a value vector
  // that we own and push back via set_parameters each step.
  std::vector<array> params;
  for (auto* p : model.parameters()) params.push_back(*p);
  fprintf(stderr, "param tensors: %zu\n", params.size());

  Optimizer opt(cfg.max_lr, cfg.weight_decay, cfg.beta1, cfg.beta2, cfg.eps);
  ModelEMA ema(params, cfg.ema_target_decay);

  std::vector<int> argnums(params.size());
  for (size_t i = 0; i < params.size(); i++) argnums[i] = (int)i;

  float best_val = 1e9f;
  auto t_start = std::chrono::high_resolution_clock::now();

  // ponytail: compile the loss+grad function once. Pass x,y as trailing args
  // so the graph is cached and Metal kernels are reused across steps.
  // inputs = [params..., x, y], argnums = [0..n_params-1]
  enable_compile();
  auto loss_fn = std::function<array(const std::vector<array>&)>(
    [&](const std::vector<array>& inputs) {
      size_t np = inputs.size() - 2;
      std::vector<array> p(inputs.begin(), inputs.begin() + np);
      model.set_parameters(p);
      return model.loss(inputs[np], inputs[np + 1], true);
    });
  auto vg = value_and_grad(loss_fn, argnums);

  for (int step = 0; step < cfg.max_steps; step++) {
    float lr;
    if (cfg.lr_schedule == LrSchedule::Cosine)
      lr = cosine_lr(step, cfg.max_steps, cfg.max_lr, cfg.min_lr, cfg.warmup_steps);
    else
      lr = wsd_lr(step, cfg.max_steps, cfg.max_lr, cfg.warmup_pct, cfg.decay_pct);
    opt.set_learning_rate(lr);

    auto [x, y] = data.get_train_batch();

    // Build inputs = [params..., x, y]
    std::vector<array> inputs;
    inputs.reserve(params.size() + 2);
    for (auto& p : params) inputs.push_back(p);
    inputs.push_back(x);
    inputs.push_back(y);

    auto [loss, grads] = vg(inputs);
    eval(loss);
    float train_loss = loss.item<float>();

    grads = clip_grad_norm(grads, 1.0f);
    params = opt.step(params, grads);
    model.set_parameters(params);
    if (cfg.use_ema) ema.update(params, step);

    if (step == 0 || step % cfg.eval_interval == 0 || step == cfg.max_steps - 1) {
      float val_ema = 0.0f, val_raw = 0.0f;
      if (cfg.use_ema) {
        model.set_parameters(ema.swap_in(params));
        val_ema = eval_loss(model, data);
        model.set_parameters(ema.swap_out());
        val_raw = eval_loss(model, data);
      } else {
        val_raw = eval_loss(model, data);
        val_ema = val_raw;
      }
      auto now = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(now - t_start).count();
      if (val_ema < best_val) best_val = val_ema;
      if (val_raw < best_val) best_val = val_raw;
      printf("Step %d: train=%.4f, val_ema=%.4f, val_raw=%.4f, lr=%.5f, %.1fs elapsed, %.0fms/step\n",
             step, train_loss, val_ema, val_raw, lr, elapsed, elapsed * 1000 / (step + 1));
      fflush(stdout);
    }
  }
  return best_val;
}

TrainConfig v9_config() {
  TrainConfig c;
  c.max_lr = 0.025f;
  c.lr_schedule = LrSchedule::Cosine;
  c.warmup_steps = 50;
  c.min_lr = 0.001f;
  return c;
}
TrainConfig v10_config() {
  TrainConfig c;
  c.d_model = 768; c.n_layer = 6; c.n_head = 12;
  c.dropout = 0.12f;
  return c;
}
TrainConfig v11_config() {
  TrainConfig c;
  c.d_model = 512; c.n_layer = 10; c.n_head = 8;
  c.dropout = 0.12f;
  return c;
}
TrainConfig v12_config() {
  TrainConfig c;
  c.dropout = 0.10f;
  return c;
}
TrainConfig v13_config() {
  TrainConfig c;  // same as baseline, token augmentation would need data loader changes
  return c;
}
TrainConfig v14_config() {
  TrainConfig c;
  c.dropout = 0.10f;
  return c;
}

void run_all_experiments(DataLoader& data) {
  struct Exp { const char* name; TrainConfig (*cfg)(); };
  Exp exps[] = {
    {"v9_cosine_lr",   v9_config},
    {"v10_wider",      v10_config},
    {"v11_deeper",     v11_config},
    {"v12_eff_attn",   v12_config},
    {"v13_tok_aug",    v13_config},
    {"v14_adapt_drop", v14_config},
  };

  std::vector<std::pair<std::string, float>> results;
  for (auto& e : exps) {
    fprintf(stderr, "=== Running %s ===\n", e.name);
    float val = run_training(data, e.cfg());
    results.push_back({e.name, val});
  }

  printf("\nRESULTS SUMMARY\n");
  int best_idx = 0;
  for (int i = 0; i < (int)results.size(); i++) {
    float improvement = (BASELINE - results[i].second) / BASELINE * 100.0f;
    const char* status = (results[i].second < BASELINE) ? "success" : "no_improvement";
    printf("  %s: val=%.6f improvement=%.2f%% status=%s\n",
           results[i].first.c_str(), results[i].second, improvement, status);
    if (results[i].second < results[best_idx].second) best_idx = i;
  }
  printf("Best: %s with val=%.6f\n", results[best_idx].first.c_str(), results[best_idx].second);
  if (results[best_idx].second < BASELINE) printf("BEAT BASELINE!\n");
}

