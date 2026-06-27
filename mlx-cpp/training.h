#pragma once
#include <mlx/mlx.h>
#include <vector>
#include <string>
#include "data.h"

enum class LrSchedule { WSD, Cosine };

struct TrainConfig {
  int max_steps = 945;
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
  LrSchedule lr_schedule = LrSchedule::WSD;
  float min_lr = 0.001f;      // for cosine
  int warmup_steps = 50;      // for cosine (override warmup_pct)
};

constexpr float BASELINE = 0.974555f;

// WSD lambda schedule: warmup -> constant -> linear decay to 0.
float wsd_lr(int step, int max_steps, float max_lr, float warmup_pct, float decay_pct);

// Cosine schedule with linear warmup.
float cosine_lr(int step, int max_steps, float max_lr, float min_lr, int warmup_steps);

float run_training(DataLoader& data, const TrainConfig& cfg);

TrainConfig v9_config();
TrainConfig v10_config();
TrainConfig v11_config();
TrainConfig v12_config();
TrainConfig v13_config();
TrainConfig v14_config();

void run_all_experiments(DataLoader& data);
