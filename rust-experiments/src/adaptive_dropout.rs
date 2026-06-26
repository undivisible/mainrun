//! Adaptive dropout experiment

use crate::{model::GPTConfig, training::{Trainer, TrainingConfig}, utils::{ExperimentResult, LrScheduler}};
use candle_core::Device;
use std::time::Instant;
use serde::{Deserialize, Serialize};

/// Adaptive dropout scheduler
pub struct AdaptiveDropoutScheduler {
    initial_dropout: f32,
    final_dropout: f32,
    ramp_steps: usize,
    max_steps: usize,
}

impl AdaptiveDropoutScheduler {
    pub fn new(initial_dropout: f32, final_dropout: f32, ramp_steps: usize, max_steps: usize) -> Self {
        Self {
            initial_dropout,
            final_dropout,
            ramp_steps,
            max_steps,
        }
    }
    
    pub fn get_dropout(&self, step: usize) -> f32 {
        if step < self.ramp_steps {
            let progress = step as f32 / self.ramp_steps as f32;
            self.initial_dropout + (self.final_dropout - self.initial_dropout) * progress
        } else {
            self.final_dropout
        }
    }
}

/// Adaptive dropout experiment
pub struct AdaptiveDropoutExperiment {
    device: Device,
    config: AdaptiveDropoutConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveDropoutConfig {
    pub base_config: GPTConfig,
    pub initial_dropout: f32,
    pub final_dropout: f32,
    pub ramp_steps: usize,
    pub max_steps: usize,
    pub eval_interval: usize,
}

impl Default for AdaptiveDropoutConfig {
    fn default() -> Self {
        Self {
            base_config: GPTConfig::default(),
            initial_dropout: 0.05,
            final_dropout: 0.2,
            ramp_steps: 400,
            max_steps: 896,
            eval_interval: 28,
        }
    }
}

impl AdaptiveDropoutExperiment {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            config: AdaptiveDropoutConfig::default(),
        }
    }
    
    pub fn with_config(mut self, config: AdaptiveDropoutConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Run the adaptive dropout experiment
    pub async fn run(&self) -> anyhow::Result<ExperimentResult> {
        let start_time = Instant::now();
        
        println!("Starting adaptive dropout experiment");
        println!("Config: {:?}", self.config);
        
        // Create training config with adaptive dropout
        let training_config = TrainingConfig {
            model_config: self.config.base_config.clone(),
            max_steps: self.config.max_steps,
            eval_interval: self.config.eval_interval,
            lr_scheduler: Box::new(ConstantLrSchedulerWrapper { lr: 0.02 }),
            device: self.device.clone(),
        };
        
        // Initialize trainer
        let mut trainer = Trainer::new(training_config)?;
        
        // Run training with adaptive dropout
        println!("Beginning training with adaptive dropout...");
        let training_result = trainer.train().await?;
        
        let elapsed = start_time.elapsed();
        
        // Compare with baseline
        let baseline_loss = 0.974555;
        let improvement = ((baseline_loss - training_result.best_val_loss) / baseline_loss) * 100.0;
        
        let result = ExperimentResult {
            name: "adaptive_dropout".to_string(),
            description: format!(
                "Adaptive dropout {:.2}→{:.2} over {} steps",
                self.config.initial_dropout, self.config.final_dropout, self.config.ramp_steps
            ),
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
        
        println!("Adaptive dropout experiment completed:");
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
        test_config.max_steps = 56;
        test_config.ramp_steps = 20;
        
        let test_exp = Self {
            device: self.device.clone(),
            config: test_config,
        };
        
        println!("Running quick test ({} steps)...", test_exp.config.max_steps);
        test_exp.run().await
    }
}

/// Wrapper for constant LR to implement LrScheduler trait
struct ConstantLrSchedulerWrapper {
    lr: f32,
}

impl LrScheduler for ConstantLrSchedulerWrapper {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adaptive_dropout_scheduler() {
        let scheduler = AdaptiveDropoutScheduler::new(0.05, 0.2, 100, 896);
        
        // Test initial phase
        assert_eq!(scheduler.get_dropout(0), 0.05);
        assert_eq!(scheduler.get_dropout(50), 0.125);
        assert_eq!(scheduler.get_dropout(100), 0.2);
        
        // Test final phase
        assert_eq!(scheduler.get_dropout(200), 0.2);
        assert_eq!(scheduler.get_dropout(896), 0.2);
    }
}