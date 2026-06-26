//! Basic cosine learning rate experiment with forward pass only

use crate::{
    model::GPTConfig, 
    basic_training::{BasicTrainer, BasicTrainingConfig}, 
    utils::{ExperimentResult, LrScheduler},
    simple_data::load_simple_data,
};
use candle_core::Device;
use std::time::Instant;
use serde::{Deserialize, Serialize};

/// Basic cosine LR experiment
pub struct BasicCosineLRExperiment {
    device: Device,
    config: BasicCosineLRConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicCosineLRConfig {
    pub base_config: GPTConfig,
    pub max_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub eval_interval: usize,
    pub learning_rate: f32,
}

impl Default for BasicCosineLRConfig {
    fn default() -> Self {
        Self {
            base_config: GPTConfig::default(),
            max_lr: 0.025,
            min_lr: 0.001,
            warmup_steps: 100,
            max_steps: 896,
            eval_interval: 28,
            learning_rate: 0.01,
        }
    }
}

impl BasicCosineLRExperiment {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            config: BasicCosineLRConfig::default(),
        }
    }
    
    pub fn with_config(mut self, config: BasicCosineLRConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Run the basic cosine LR experiment
    pub fn run(&self) -> anyhow::Result<ExperimentResult> {
        let start_time = Instant::now();
        
        println!("Starting BASIC cosine LR experiment");
        println!("Config: {:?}", self.config);
        
        // Load simple data with dynamic vocab size
        let actual_vocab_size = self.config.base_config.vocab_size.min(1000); // Limit to reasonable size
        let (data_loader, tokenizer) = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(load_simple_data(
                actual_vocab_size,
                10000, // num_titles
                0.1,   // val_frac
                self.device.clone(),
            ))
        })?;
        
        // Create training config with cosine LR and correct vocab size
        let mut model_config = self.config.base_config.clone();
        model_config.vocab_size = tokenizer.vocab_size();
        
        let training_config = BasicTrainingConfig {
            model_config,
            max_steps: self.config.max_steps,
            eval_interval: self.config.eval_interval,
            lr_scheduler: Box::new(BasicCosineLrSchedulerWrapper {
                max_lr: self.config.max_lr,
                min_lr: self.config.min_lr,
                warmup_steps: self.config.warmup_steps,
                max_steps: self.config.max_steps,
            }),
            device: self.device.clone(),
            learning_rate: self.config.learning_rate,
        };
        
        // Initialize basic trainer
        let mut trainer = BasicTrainer::new(training_config, data_loader, tokenizer)?;
        
        // Run basic training
        println!("Beginning BASIC training with cosine LR scheduling...");
        let training_result = trainer.train()?;
        
        let elapsed = start_time.elapsed();
        
        // Test generation
        let sample = trainer.generate_sample("The future of AI is", 50)?;
        println!("Sample generation: {}", sample);
        
        // Compare with baseline
        let baseline_loss = 0.974555;
        let improvement = ((baseline_loss - training_result.best_val_loss) / baseline_loss) * 100.0;
        
        let result = ExperimentResult {
            name: "basic_cosine_lr".to_string(),
            description: "BASIC cosine learning rate with warmup (0.025→0.001)".to_string(),
            best_val_loss: training_result.best_val_loss,
            baseline_loss,
            improvement_percent: improvement,
            training_time: elapsed,
            steps_completed: training_result.steps_completed,
            config: serde_json::to_value(&self.config)?,
            status: if training_result.best_val_loss < baseline_loss {
                "success".to_string()
            } else {
                "no_improvement".to_string()
            },
        };
        
        println!("BASIC cosine LR experiment completed:");
        println!("  Best validation loss: {:.6}", result.best_val_loss);
        println!("  Improvement: {:.2}%", result.improvement_percent);
        println!("  Training time: {:?}", elapsed);
        
        if result.best_val_loss < baseline_loss {
            println!("🎉 NEW BEST RESULT!");
        } else {
            println!("No improvement over baseline");
        }
        
        Ok(result)
    }
    
    /// Quick test run with fewer steps
    pub fn quick_test(&self) -> anyhow::Result<ExperimentResult> {
        let mut test_config = self.config.clone();
        test_config.max_steps = 56; // 2 evaluation intervals
        test_config.warmup_steps = 10;
        
        let test_exp = Self {
            device: self.device.clone(),
            config: test_config,
        };
        
        println!("Running BASIC quick test ({} steps)...", test_exp.config.max_steps);
        test_exp.run()
    }
}

/// Wrapper for basic cosine LR to implement LrScheduler trait
struct BasicCosineLrSchedulerWrapper {
    max_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    max_steps: usize,
}

impl LrScheduler for BasicCosineLrSchedulerWrapper {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            return self.max_lr * step as f32 / self.warmup_steps as f32;
        }
        
        let progress = (step - self.warmup_steps) as f32 / (self.max_steps - self.warmup_steps) as f32;
        self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_cosine_lr_schedule() {
        let scheduler = BasicCosineLrSchedulerWrapper {
            max_lr: 0.025,
            min_lr: 0.001,
            warmup_steps: 100,
            max_steps: 896,
        };
        
        // Test warmup phase
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(50), 0.0125);
        assert_eq!(scheduler.get_lr(100), 0.025);
        
        // Test decay phase
        let mid_lr = scheduler.get_lr(498); // Middle of training
        assert!(mid_lr < 0.025 && mid_lr > 0.001);
        
        // Test final phase
        let final_lr = scheduler.get_lr(896);
        assert!((final_lr - 0.001).abs() < 1e-6);
    }
}