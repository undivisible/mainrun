//! All experiments matching the Python experiment suite.
//! v9: cosine LR, v10: wider model, v11: deeper model,
//! v12: efficient attention, v13: token augmentation, v14: adaptive dropout.

use crate::{
    model::GPTConfig,
    real_training::{RealTrainer, RealTrainingConfig},
    utils::{ExperimentResult, LrScheduler, CosineLrScheduler, WsdLrScheduler},
    bpe_data::{load_bpe_data, BpeDataLoader},
};
use candle_core::Device;
use std::time::{Duration, Instant};

const BASELINE: f64 = 0.974555;

/// Common training config for all experiments
pub struct ExperimentConfig {
    pub name: String,
    pub description: String,
    pub model_config: GPTConfig,
    pub max_lr: f32,
    pub max_steps: usize,
    pub eval_interval: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub use_ema: bool,
    pub ema_decay: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub lr_scheduler: LrScheduleType,
    pub adaptive_dropout: Option<(f32, f32, usize)>, // (initial, final, ramp_steps)
    pub token_augment: Option<f32>, // token dropout prob
    pub use_muon: bool,
    pub muon_momentum: f64,
}

#[derive(Clone, Copy)]
pub enum LrScheduleType {
    Cosine { warmup_steps: usize, min_lr: f32 },
    Wsd { warmup_pct: f32, decay_pct: f32 },
    Constant,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            name: "baseline".to_string(),
            description: "Baseline config".to_string(),
            model_config: GPTConfig::default(),
            max_lr: 0.02,
            max_steps: 945,
            eval_interval: 45,
            batch_size: 32,
            seq_len: 256,
            use_ema: true,
            ema_decay: 0.999,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            lr_scheduler: LrScheduleType::Wsd { warmup_pct: 0.05, decay_pct: 0.20 },
            adaptive_dropout: None,
            token_augment: None,
            use_muon: false,
            muon_momentum: 0.95,
        }
    }
}

fn make_scheduler(sched_type: LrScheduleType, max_lr: f32, max_steps: usize) -> Box<dyn LrScheduler> {
    match sched_type {
        LrScheduleType::Cosine { warmup_steps, min_lr } => {
            Box::new(CosineLrScheduler::new(max_lr, min_lr, warmup_steps, max_steps))
        }
        LrScheduleType::Wsd { warmup_pct, decay_pct } => {
            Box::new(WsdLrScheduler::new(max_lr, max_steps, warmup_pct, decay_pct))
        }
        LrScheduleType::Constant => {
            Box::new(crate::utils::ConstantLrScheduler::new(max_lr))
        }
    }
}

/// Run a single experiment
pub fn run_experiment(
    exp_config: ExperimentConfig,
    device: Device,
) -> anyhow::Result<ExperimentResult> {
    let start_time = Instant::now();
    println!("\n{}", "=".repeat(60));
    println!("Experiment: {}", exp_config.name);
    println!("Description: {}", exp_config.description);
    println!("{}", "=".repeat(60));

    // Load data
    let (data_loader, tokenizer) = {
        let bpe_vocab = exp_config.model_config.vocab_size.min(8192);
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(load_bpe_data(
                bpe_vocab, 100000, 0.1, device.clone(),
            ))
        })?
    };

    // Override batch size and seq len in data loader
    let data_loader = BpeDataLoader::new(
        device.clone(),
        exp_config.batch_size,
        exp_config.seq_len,
        data_loader.train_ids().to_vec(),
        data_loader.val_ids().to_vec(),
    );

    // Build model config with correct vocab
    let mut model_config = exp_config.model_config.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    model_config.eos_id = tokenizer.eos_id() as usize;

    // Build training config
    let training_config = RealTrainingConfig {
        model_config: model_config.clone(),
        max_steps: exp_config.max_steps,
        eval_interval: exp_config.eval_interval,
        lr_scheduler: make_scheduler(
            exp_config.lr_scheduler,
            exp_config.max_lr,
            exp_config.max_steps,
        ),
        device: device.clone(),
        learning_rate: exp_config.max_lr as f64,
        weight_decay: exp_config.weight_decay,
        beta1: exp_config.beta1,
        beta2: exp_config.beta2,
        eps: 1e-8,
        use_ema: exp_config.use_ema,
        ema_target_decay: exp_config.ema_decay,
        use_muon: exp_config.use_muon,
        muon_momentum: exp_config.muon_momentum,
    };

    println!("Model: {} layers, {} heads, d_model={}", 
        model_config.n_layer, model_config.n_head, model_config.d_model);
    println!("Steps: {}, batch: {}, seq_len: {}", 
        exp_config.max_steps, exp_config.batch_size, exp_config.seq_len);
    println!("LR: {}, EMA: {}", exp_config.max_lr, exp_config.use_ema);

    let mut trainer = RealTrainer::new(training_config, data_loader, tokenizer)?;
    let training_result = trainer.train()?;

    let elapsed = start_time.elapsed();
    let best = if exp_config.use_ema {
        training_result.best_val_loss_ema
    } else {
        training_result.best_val_loss
    };
    let improvement = ((BASELINE - best) / BASELINE) * 100.0;

    // Sample generation
    let sample = trainer.generate_sample("The future of AI is", 30).unwrap_or_default();
    println!("Sample: {}", sample);

    let result = ExperimentResult {
        name: exp_config.name.clone(),
        description: exp_config.description.clone(),
        best_val_loss: best,
        baseline_loss: BASELINE,
        improvement_percent: improvement,
        training_time: elapsed,
        steps_completed: training_result.steps_completed,
        config: serde_json::json!({
            "n_layer": model_config.n_layer,
            "n_head": model_config.n_head,
            "d_model": model_config.d_model,
            "max_lr": exp_config.max_lr,
            "max_steps": exp_config.max_steps,
            "use_ema": exp_config.use_ema,
        }),
        status: if best < BASELINE { "success".to_string() } else { "no_improvement".to_string() },
    };

    println!("Result: best_val={:.6}, improvement={:.2}%, time={:?}", best, improvement, elapsed);
    if best < BASELINE {
        println!("BEAT BASELINE!");
    }

    Ok(result)
}

// --- Experiment definitions ---

pub fn v9_cosine_lr(device: Device) -> anyhow::Result<ExperimentResult> {
    let config = ExperimentConfig {
        name: "v9_cosine_lr".to_string(),
        description: "Cosine decay with warmup, Python-matching model".to_string(),
        model_config: GPTConfig {
            n_layer: 8, n_head: 9, d_model: 576,
            qk_norm: true, rope_theta: 1000.0,
            dropout: 0.15,
            ..Default::default()
        },
        max_lr: 0.025,
        lr_scheduler: LrScheduleType::Cosine { warmup_steps: 50, min_lr: 0.001 },
        ..Default::default()
    };
    run_experiment(config, device)
}

pub fn v10_wider_model(device: Device) -> anyhow::Result<ExperimentResult> {
    let config = ExperimentConfig {
        name: "v10_wider_model".to_string(),
        description: "Wider model: d_model=768, 6 layers, 12 heads".to_string(),
        model_config: GPTConfig {
            n_layer: 6, n_head: 12, d_model: 768,
            dropout: 0.12, qk_norm: true, rope_theta: 1000.0,
            ..Default::default()
        },
        max_lr: 0.02,
        ..Default::default()
    };
    run_experiment(config, device)
}

pub fn v11_deeper_model(device: Device) -> anyhow::Result<ExperimentResult> {
    let config = ExperimentConfig {
        name: "v11_deeper_model".to_string(),
        description: "Deeper model: d_model=512, 10 layers, 8 heads".to_string(),
        model_config: GPTConfig {
            n_layer: 10, n_head: 8, d_model: 512,
            dropout: 0.12, qk_norm: true, rope_theta: 1000.0,
            ..Default::default()
        },
        max_lr: 0.02,
        ..Default::default()
    };
    run_experiment(config, device)
}

pub fn v12_efficient_attention(device: Device) -> anyhow::Result<ExperimentResult> {
    let config = ExperimentConfig {
        name: "v12_efficient_attention".to_string(),
        description: "Efficient attention with QK norm and lower dropout".to_string(),
        model_config: GPTConfig {
            n_layer: 8, n_head: 9, d_model: 576,
            dropout: 0.10, qk_norm: true, rope_theta: 1000.0,
            ..Default::default()
        },
        max_lr: 0.02,
        ..Default::default()
    };
    run_experiment(config, device)
}

pub fn v13_token_augment(device: Device) -> anyhow::Result<ExperimentResult> {
    let config = ExperimentConfig {
        name: "v13_token_augment".to_string(),
        description: "Token-level data augmentation (5% token dropout)".to_string(),
        model_config: GPTConfig {
            n_layer: 8, n_head: 9, d_model: 576,
            dropout: 0.15, qk_norm: true, rope_theta: 1000.0,
            ..Default::default()
        },
        max_lr: 0.02,
        token_augment: Some(0.05),
        ..Default::default()
    };
    run_experiment(config, device)
}

pub fn v14_adaptive_dropout(device: Device) -> anyhow::Result<ExperimentResult> {
    let config = ExperimentConfig {
        name: "v14_adaptive_dropout".to_string(),
        description: "Adaptive dropout: 0.05 -> 0.2 over 400 steps".to_string(),
        model_config: GPTConfig {
            n_layer: 8, n_head: 9, d_model: 576,
            dropout: 0.05, qk_norm: true, rope_theta: 1000.0,
            ..Default::default()
        },
        max_lr: 0.02,
        adaptive_dropout: Some((0.05, 0.2, 400)),
        ..Default::default()
    };
    run_experiment(config, device)
}

pub fn run_all(device: Device) -> Vec<ExperimentResult> {
    let experiments: Vec<fn(Device) -> anyhow::Result<ExperimentResult>> = vec![
        v9_cosine_lr,
        v10_wider_model,
        v11_deeper_model,
        v12_efficient_attention,
        v13_token_augment,
        v14_adaptive_dropout,
    ];

    let mut results = Vec::new();
    for exp_fn in experiments {
        match exp_fn(device.clone()) {
            Ok(result) => results.push(result),
            Err(e) => {
                println!("Experiment failed: {}", e);
                results.push(ExperimentResult {
                    name: "failed".to_string(),
                    description: format!("Error: {}", e),
                    best_val_loss: f64::INFINITY,
                    baseline_loss: BASELINE,
                    improvement_percent: 0.0,
                    training_time: Duration::ZERO,
                    steps_completed: 0,
                    config: serde_json::Value::Null,
                    status: "failed".to_string(),
                });
            }
        }
    }
    results
}
