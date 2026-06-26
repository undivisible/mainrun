//! Cosine learning rate scheduling experiment

use crate::{model::GPTConfig, training::{Trainer, TrainingConfig}, utils::{ExperimentResult, LrScheduler}};
use candle_core::Device;
use std::time::Instant;
use serde::{Deserialize, Serialize};

/// Cosine learning rate with warmup
pub fn cosine_lr_with_warmup(
    step: usize,
    max_steps: usize,
    warmup_steps: usize,
    max_lr: f32,
    min_lr: f32,
) -> f32 {
    if step < warmup_steps {
        return max_lr * step as f32 / warmup_steps as f32;
    }
    
    let progress = (step - warmup_steps) as f32 / (max_steps - warmup_steps) as f32;
    min_lr + (max_lr - min_lr) * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())
}

/// Cosine LR scheduling experiment
pub struct CosineLRExperiment {
    device: Device,
    config: CosineLRConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosineLRConfig {
    pub base_config: GPTConfig,
    pub max_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub eval_interval: usize,
}

impl Default for CosineLRConfig {
    fn default() -> Self {
        Self {
            base_config: GPTConfig::default(),
            max_lr: 0.025,
            min_lr: 0.001,
            warmup_steps: 100,
            max_steps: 896,
            eval_interval: 28,
        }
    }
}

impl CosineLRExperiment {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            config: CosineLRConfig::default(),
        }
    }
    
    pub fn with_config(mut self, config: CosineLRConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Run the cosine LR experiment
    pub async fn run(&self) -> anyhow::Result<ExperimentResult> {
        let start_time = Instant::now();
        
        println!("Starting cosine LR experiment");
        println!("Config: {:?}", self.config);
        
        // Create training config with cosine LR
        let training_config = TrainingConfig {
            model_config: self.config.base_config.clone(),
            max_steps: self.config.max_steps,
            eval_interval: self.config.eval_interval,
            lr_scheduler: Box::new(CosineLrSchedulerWrapper {
                max_lr: self.config.max_lr,
                min_lr: self.config.min_lr,
                warmup_steps: self.config.warmup_steps,
                max_steps: self.config.max_steps,
            }),
            device: self.device.clone(),
        };
        
        // Initialize trainer
        let mut trainer = Trainer::new(training_config)?;
        
        // Run training
        println!("Beginning training with cosine LR scheduling...");
        let training_result = trainer.train().await?;
        
        let elapsed = start_time.elapsed();
        
        // Compare with baseline
        let baseline_loss = 0.974555;
        let improvement = ((baseline_loss - training_result.best_val_loss) / baseline_loss) * 100.0;
        
        let result = ExperimentResult {
            name: "cosine_lr".to_string(),
            description: "Cosine learning rate with warmup (0.025→0.001)".to_string(),
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
        
        println!("Cosine LR experiment completed:");
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
    pub async fn quick_test(&self) -> anyhow::Result<ExperimentResult> {
        let mut test_config = self.config.clone();
        test_config.max_steps = 56; // 2 evaluation intervals
        test_config.warmup_steps = 10;
        
        let test_exp = Self {
            device: self.device.clone(),
            config: test_config,
        };
        
        println!("Running quick test ({} steps)...", test_exp.config.max_steps);
        test_exp.run().await
    }
}

/// Wrapper for cosine LR to implement LrScheduler trait
struct CosineLrSchedulerWrapper {
    max_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    max_steps: usize,
}

impl LrScheduler for CosineLrSchedulerWrapper {
    fn get_lr(&self, step: usize) -> f32 {
        cosine_lr_with_warmup(
            step,
            self.max_steps,
            self.warmup_steps,
            self.max_lr,
            self.min_lr,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_lr_schedule() {
        let max_lr = 0.025;
        let min_lr = 0.001;
        let warmup = 100;
        let max_steps = 896;
        
        // Test warmup phase
        assert_eq!(cosine_lr_with_warmup(0, max_steps, warmup, max_lr, min_lr), 0.0);
        assert_eq!(cosine_lr_with_warmup(50, max_steps, warmup, max_lr, min_lr), max_lr * 0.5);
        assert_eq!(cosine_lr_with_warmup(100, max_steps, warmup, max_lr, min_lr), max_lr);
        
        // Test decay phase
        let mid_lr = cosine_lr_with_warmup(498, max_steps, warmup, max_lr, min_lr); // Middle of training
        assert!(mid_lr < max_lr && mid_lr > min_lr);
        
        // Test final phase
        let final_lr = cosine_lr_with_warmup(896, max_steps, warmup, max_lr, min_lr);
        assert!((final_lr - min_lr).abs() < 1e-6);
    }
}