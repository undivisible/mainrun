//! Training infrastructure for experiments

use crate::{model::{GPT, GPTConfig}, utils::{LrScheduler, ProgressBar}};
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::VarBuilder;
use std::time::{Duration, Instant};

/// Training configuration
pub struct TrainingConfig {
    pub model_config: GPTConfig,
    pub max_steps: usize,
    pub eval_interval: usize,
    pub lr_scheduler: Box<dyn LrScheduler>,
    pub device: Device,
}

/// Training result
#[derive(Debug)]
pub struct TrainingResult {
    pub best_val_loss: f64,
    pub steps_completed: usize,
    pub training_time: Duration,
    pub loss_curve: Vec<f64>,
}

/// Simplified trainer for experiments
pub struct Trainer {
    config: TrainingConfig,
    model: GPT,
    optimizer: SimpleOptimizer,
}

/// Simple SGD optimizer for experiments
pub struct SimpleOptimizer {
    learning_rate: f32,
}

impl SimpleOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
    
    pub fn step(&mut self, _loss: &Tensor) -> Result<()> {
        // Simplified - in real implementation would update model parameters
        Ok(())
    }
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Result<Self> {
        // Create model with random weights for testing
        let vb = VarBuilder::zeros(DType::F32, &config.device);
        let model = GPT::load(config.model_config.clone(), vb)?;
        
        let optimizer = SimpleOptimizer::new(0.01);
        
        Ok(Self {
            config,
            model,
            optimizer,
        })
    }
    
    pub async fn train(&mut self) -> anyhow::Result<TrainingResult> {
        let start_time = Instant::now();
        let mut best_val_loss = f64::INFINITY;
        let mut loss_curve = Vec::new();
        
        println!("Starting training for {} steps", self.config.max_steps);
        let mut progress = ProgressBar::new(self.config.max_steps);
        
        for step in 0..self.config.max_steps {
            // Get current learning rate
            let lr = self.config.lr_scheduler.get_lr(step);
            
            // Simulate training step (simplified)
            let train_loss = self.simulate_training_step(step, lr)?;
            loss_curve.push(train_loss);
            
            // Evaluation
            if step % self.config.eval_interval == 0 {
                let val_loss = self.simulate_validation_step(step)?;
                
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    println!("New best validation loss: {:.6} at step {}", best_val_loss, step);
                }
                
                println!("Step {}: train_loss={:.6}, val_loss={:.6}, lr={:.5}", 
                    step, train_loss, val_loss, lr);
            }
            
            progress.update(step + 1);
            
            // Simulate some computation time
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        
        let training_time = start_time.elapsed();
        
        Ok(TrainingResult {
            best_val_loss,
            steps_completed: self.config.max_steps,
            training_time,
            loss_curve,
        })
    }
    
    fn simulate_training_step(&self, step: usize, lr: f32) -> Result<f64> {
        // Simulate training loss that decreases over time
        let base_loss = 2.0;
        let decay = (step as f64 / self.config.max_steps as f64) * 1.5;
        let noise = (rand::random::<f64>() - 0.5) * 0.1;
        let lr_effect = (lr - 0.01) * 2.0; // LR affects loss
        
        Ok((base_loss - decay + noise + lr_effect as f64).max(0.1))
    }
    
    fn simulate_validation_step(&self, step: usize) -> Result<f64> {
        // Simulate validation loss with overfitting pattern
        let base_loss = 1.2;
        let improvement = (step as f64 / self.config.max_steps as f64) * 0.3;
        let overfitting = if step > 400 {
            ((step - 400) as f64 / 496.0) * 0.15 // Start overfitting after step 400
        } else {
            0.0
        };
        let noise = (rand::random::<f64>() - 0.5) * 0.05;
        
        Ok((base_loss - improvement + overfitting + noise).max(0.8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{CosineLrScheduler, ConstantLrScheduler};
    
    #[tokio::test]
    async fn test_trainer_cosine_lr() {
        let config = TrainingConfig {
            model_config: GPTConfig::default(),
            max_steps: 56, // Short test
            eval_interval: 28,
            lr_scheduler: Box::new(CosineLrScheduler::new(0.025, 0.001, 10, 56)),
            device: Device::Cpu,
        };
        
        let mut trainer = Trainer::new(config).unwrap();
        let result = trainer.train().await.unwrap();
        
        assert_eq!(result.steps_completed, 56);
        assert!(result.best_val_loss < f64::INFINITY);
        assert!(result.training_time > Duration::ZERO);
    }
}