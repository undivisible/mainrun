//! Simple training implementation without complex optimizer dependencies

use crate::{
    model::{GPT, GPTConfig}, 
    utils::{LrScheduler, ProgressBar},
    simple_data::{SimpleDataLoader, SimpleTokenizer},
};
use candle_core::{Device, Result, Tensor, DType, IndexOp, Module, Var};
use candle_nn::{VarBuilder};
use std::time::{Duration, Instant};

/// Simple training configuration
pub struct SimpleTrainingConfig {
    pub model_config: GPTConfig,
    pub max_steps: usize,
    pub eval_interval: usize,
    pub lr_scheduler: Box<dyn LrScheduler>,
    pub device: Device,
    pub learning_rate: f32,
}

/// Simple training result
#[derive(Debug)]
pub struct SimpleTrainingResult {
    pub best_val_loss: f64,
    pub steps_completed: usize,
    pub training_time: Duration,
    pub loss_curve: Vec<f64>,
    pub val_loss_curve: Vec<f64>,
}

/// Simple trainer with basic SGD optimization
pub struct SimpleTrainer {
    config: SimpleTrainingConfig,
    model: GPT,
    data_loader: SimpleDataLoader,
    tokenizer: SimpleTokenizer,
    vars: Vec<Var>,
}

impl SimpleTrainer {
    pub fn new(
        config: SimpleTrainingConfig,
        data_loader: SimpleDataLoader,
        tokenizer: SimpleTokenizer,
    ) -> Result<Self> {
        // Create model with real parameters
        let vb = VarBuilder::zeros(DType::F32, &config.device);
        let model = GPT::load(config.model_config.clone(), vb)?;
        
        // Get all variables from the model
        let vars = model.parameters().collect();
        
        Ok(Self {
            config,
            model,
            data_loader,
            tokenizer,
            vars,
        })
    }
    
    pub fn train(&mut self) -> anyhow::Result<SimpleTrainingResult> {
        let start_time = Instant::now();
        let mut best_val_loss = f64::INFINITY;
        let mut loss_curve = Vec::new();
        let mut val_loss_curve = Vec::new();
        
        println!("Starting simple training for {} steps", self.config.max_steps);
        let mut progress = ProgressBar::new(self.config.max_steps);
        
        for step in 0..self.config.max_steps {
            // Get current learning rate
            let lr = self.config.lr_scheduler.get_lr(step);
            
            // Training step
            let train_loss = self.training_step(lr)?;
            loss_curve.push(train_loss);
            
            // Evaluation
            if step % self.config.eval_interval == 0 {
                let val_loss = self.evaluate()?;
                val_loss_curve.push(val_loss);
                
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    println!("New best validation loss: {:.6} at step {}", best_val_loss, step);
                }
                
                println!("Step {}: train_loss={:.6}, val_loss={:.6}, lr={:.5}", 
                    step, train_loss, val_loss, lr);
            }
            
            progress.update(step + 1);
        }
        
        let training_time = start_time.elapsed();
        
        Ok(SimpleTrainingResult {
            best_val_loss,
            steps_completed: self.config.max_steps,
            training_time,
            loss_curve,
            val_loss_curve,
        })
    }
    
    /// Single training step with forward/backward pass
    fn training_step(&mut self, lr: f32) -> Result<f64> {
        // Get training batch
        let (input, targets) = self.data_loader.get_train_batch()?;
        
        // Forward pass
        let logits = self.model.forward(&input)?;
        
        // Compute cross-entropy loss
        let loss = self.cross_entropy_loss(&logits, &targets)?;
        
        // Backward pass
        let grads = loss.backward()?;
        
        // Simple SGD update
        for var in &self.vars {
            if let Some(grad) = var.grad() {
                let new_val = var.tensor().sub(&(grad * lr)?)?;
                var.set(&new_val)?;
            }
        }
        
        // Clear gradients
        for var in &self.vars {
            var.set_grad(&Tensor::zeros_like(var.tensor())?)?;
        }
        
        Ok(loss.to_vec0::<f32>()? as f64)
    }
    
    /// Evaluate on validation set
    fn evaluate(&mut self) -> Result<f64> {
        self.data_loader.reset_val_ptr();
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        // Evaluate on a few batches for speed
        for _ in 0..5 {
            let (input, targets) = self.data_loader.get_val_batch()?;
            
            // Forward pass
            let logits = self.model.forward(&input)?;
            
            // Compute loss
            let loss = self.cross_entropy_loss(&logits, &targets)?;
            total_loss += loss.to_vec0::<f32>()? as f64;
            num_batches += 1;
        }
        
        Ok(total_loss / num_batches as f64)
    }
    
    /// Compute cross-entropy loss
    fn cross_entropy_loss(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Reshape for loss computation
        let (batch_size, seq_len, vocab_size) = logits.dims3()?;
        let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
        let targets_flat = targets.reshape((batch_size * seq_len,))?;
        
        // Compute cross entropy
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
        Ok(loss)
    }
    
    /// Generate text sample for evaluation
    pub fn generate_sample(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let mut tokens = self.tokenizer.encode(prompt)?;
        let eos_id = self.tokenizer.eos_id();
        
        for _ in 0..max_tokens {
            // Create input tensor
            let input = Tensor::from_vec(
                tokens.clone(), 
                (1, tokens.len()), 
                &self.config.device,
            )?;
            
            // Forward pass
            let logits = self.model.forward(&input)?;
            
            // Get next token (sample from distribution)
            let next_token_logits = logits.i((0, -1))?; // Last token
            let probs = candle_nn::ops::softmax(&next_token_logits, 0)?;
            let next_token = self.sample_from_distribution(&probs)?;
            
            tokens.push(next_token);
            
            // Stop if EOS
            if next_token == eos_id {
                break;
            }
        }
        
        self.tokenizer.decode(&tokens)
    }
    
    /// Sample from probability distribution
    fn sample_from_distribution(&self, probs: &Tensor) -> Result<u32> {
        let probs_vec = probs.to_vec1::<f32>()?;
        
        // Simple sampling - could use more sophisticated methods
        let mut sum = 0.0;
        let random_val: f32 = rand::random();
        
        for (i, &prob) in probs_vec.iter().enumerate() {
            sum += prob;
            if random_val <= sum {
                return Ok(i as u32);
            }
        }
        
        Ok((probs_vec.len() - 1) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{CosineLrScheduler};
    use crate::simple_data::load_simple_data;
    
    #[tokio::test]
    async fn test_simple_trainer() {
        let device = Device::Cpu;
        
        // Load data
        let (data_loader, tokenizer) = load_simple_data(1000, 0.1, device.clone()).await.unwrap();
        
        // Create training config
        let config = SimpleTrainingConfig {
            model_config: GPTConfig {
                vocab_size: tokenizer.vocab_size(),
                n_layer: 2, // Small model for testing
                n_head: 2,
                d_model: 64,
                ..Default::default()
            },
            max_steps: 10, // Very short test
            eval_interval: 5,
            lr_scheduler: Box::new(CosineLrScheduler::new(0.01, 0.001, 2, 10)),
            device,
            learning_rate: 0.01,
        };
        
        // Create trainer
        let mut trainer = SimpleTrainer::new(config, data_loader, tokenizer).unwrap();
        
        // Train
        let result = trainer.train().unwrap();
        
        assert_eq!(result.steps_completed, 10);
        assert!(result.best_val_loss < f64::INFINITY);
        assert!(result.training_time > Duration::ZERO);
        
        // Test generation
        let sample = trainer.generate_sample("The future of", 20).unwrap();
        assert!(!sample.is_empty());
    }
}