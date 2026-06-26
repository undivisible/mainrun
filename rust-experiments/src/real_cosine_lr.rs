//! Real training experiment with WSD scheduler, EMA validation, and gradients.

use crate::{
    model::GPTConfig,
    real_training::{RealTrainer, RealTrainingConfig},
    utils::{ExperimentResult, LrScheduler, WsdLrScheduler},
    bpe_data::load_bpe_data,
};
use candle_core::Device;
use std::time::Instant;
use serde::{Deserialize, Serialize};

pub struct RealCosineLRExperiment {
    device: Device,
    config: RealCosineLRConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealCosineLRConfig {
    pub base_config: GPTConfig,
    pub max_lr: f32,
    pub max_steps: usize,
    pub eval_interval: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub use_ema: bool,
    pub ema_target_decay: f64,
    pub wsd_warmup_pct: f32,
    pub wsd_decay_pct: f32,
}

impl Default for RealCosineLRConfig {
    fn default() -> Self {
        Self {
            base_config: GPTConfig::default(),
            max_lr: 0.025,
            max_steps: 1350,
            eval_interval: 45,
            learning_rate: 0.025,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            use_ema: true,
            ema_target_decay: 0.999,
            wsd_warmup_pct: 0.05,
            wsd_decay_pct: 0.20,
        }
    }
}

impl RealCosineLRExperiment {
    pub fn new(device: Device) -> Self {
        Self { device, config: RealCosineLRConfig::default() }
    }

    pub fn with_config(mut self, config: RealCosineLRConfig) -> Self {
        self.config = config;
        self
    }

    pub fn run(&self) -> anyhow::Result<ExperimentResult> {
        let start_time = Instant::now();
        println!("Starting WSD+EMA experiment");
        println!("Config: {:?}", self.config);

        let bpe_vocab_size = self.config.base_config.vocab_size.min(8192);
        let (data_loader, tokenizer) = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(load_bpe_data(
                bpe_vocab_size, 100000, 0.1, self.device.clone(),
            ))
        })?;

        let mut model_config = self.config.base_config.clone();
        model_config.vocab_size = tokenizer.vocab_size();
        // Use Python-matching architecture: 8 layers, 9 heads, d_model=576
        model_config.n_layer = 8;
        model_config.n_head = 9;
        model_config.d_model = 576;
        model_config.block_size = 256;
        model_config.dropout = 0.15;
        model_config.rope_theta = 1000.0;
        model_config.qk_norm = true;

        let training_config = RealTrainingConfig {
            model_config,
            max_steps: self.config.max_steps,
            eval_interval: self.config.eval_interval,
            lr_scheduler: Box::new(WsdLrScheduler::new(
                self.config.max_lr,
                self.config.max_steps,
                self.config.wsd_warmup_pct,
                self.config.wsd_decay_pct,
            )),
            device: self.device.clone(),
            learning_rate: self.config.learning_rate,
            weight_decay: self.config.weight_decay,
            beta1: self.config.beta1,
            beta2: self.config.beta2,
            eps: self.config.eps,
            use_ema: self.config.use_ema,
            ema_target_decay: self.config.ema_target_decay,
            use_muon: true,
            muon_momentum: 0.95,
        };

        let mut trainer = RealTrainer::new(training_config, data_loader, tokenizer)?;
        println!("Beginning training with WSD scheduler + EMA validation...");
        let training_result = trainer.train()?;

        let elapsed = start_time.elapsed();

        let sample = trainer.generate_sample("The future of AI is", 50)?;
        println!("Sample generation: {}", sample);

        let baseline_loss = 0.974555;
        let best = if self.config.use_ema {
            training_result.best_val_loss_ema
        } else {
            training_result.best_val_loss
        };
        let improvement = ((baseline_loss - best) / baseline_loss) * 100.0;

        let result = ExperimentResult {
            name: "wsd_ema".to_string(),
            description: "WSD scheduler + EMA validation + AdamW".to_string(),
            best_val_loss: best,
            baseline_loss,
            improvement_percent: improvement,
            training_time: elapsed,
            steps_completed: training_result.steps_completed,
            config: serde_json::to_value(&self.config)?,
            status: if best < baseline_loss { "success".to_string() } else { "no_improvement".to_string() },
        };

        println!("Experiment completed:");
        println!("  Best EMA val loss: {:.6}", training_result.best_val_loss_ema);
        println!("  Best raw val loss: {:.6}", training_result.best_val_loss);
        println!("  Improvement: {:.2}%", improvement);
        println!("  Training time: {:?}", elapsed);

        if best < baseline_loss {
            println!("NEW BEST RESULT!");
        } else {
            println!("No improvement over baseline");
        }

        Ok(result)
    }

    pub fn quick_test(&self) -> anyhow::Result<ExperimentResult> {
        let mut test_config = self.config.clone();
        test_config.max_steps = 90;
        let test_exp = Self { device: self.device.clone(), config: test_config };
        println!("Running quick test ({} steps)...", test_exp.config.max_steps);
        test_exp.run()
    }
}
