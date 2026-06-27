#pragma once
#include <mlx/mlx.h>
#include <vector>
#include <string>
#include "data.h"

struct TrainConfig {
  int max_steps = 889;
  int batch_size = 32;
  int block_size = 256;
  int eval_interval = 45;
  float max_lr = 0.02f;
  float warmup_pct = 0.05f;
  float decay_pct = 0.20f;
  float weight_decay = 0.1f;
  float beta1 = 0.9f;
  float beta2 = 0.95f;
  float eps = 1e-8f;
  bool use_ema = true;
  float ema_target_decay = 0.999f;
  int n_layer = 8;
  int n_head = 9;
  int d_model = 576;
  float dropout = 0.15f;
  bool qk_norm = true;
  float rope_theta = 1000.0f;
};

constexpr float BASELINE = 1.115752f;

// WSD lambda schedule: warmup -> constant -> linear decay to 0.
float wsd_lr(int step, int max_steps, float max_lr, float warmup_pct, float decay_pct);

float run_training(DataLoader& data, const TrainConfig& cfg);

