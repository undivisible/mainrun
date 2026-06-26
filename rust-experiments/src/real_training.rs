//! Real training with gradients, weight updates, and EMA validation.

use crate::{
    model::{GPT, GPTConfig, create_gpt_model},
    utils::{LrScheduler, ProgressBar},
    bpe_data::{BpeDataLoader, BpeTokenizer},
    ema::ModelEma,
    muon::{MuonOptimizer, split_param_groups},
};
use candle_core::{Device, Result, Tensor, IndexOp};
use candle_nn::{Optimizer, AdamW, ParamsAdamW, VarMap};
use std::time::{Duration, Instant};

pub struct RealTrainingConfig {
    pub model_config: GPTConfig,
    pub max_steps: usize,
    pub eval_interval: usize,
    pub lr_scheduler: Box<dyn LrScheduler>,
    pub device: Device,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub use_ema: bool,
    pub ema_target_decay: f64,
    pub use_muon: bool,
    pub muon_momentum: f64,
}

#[derive(Debug)]
pub struct RealTrainingResult {
    pub best_val_loss: f64,
    pub best_val_loss_ema: f64,
    pub steps_completed: usize,
    pub training_time: Duration,
    pub loss_curve: Vec<f64>,
    pub val_loss_curve: Vec<f64>,
}

pub struct RealTrainer {
    config: RealTrainingConfig,
    model: GPT,
    varmap: VarMap,
    data_loader: BpeDataLoader,
    tokenizer: BpeTokenizer,
    optimizer: Option<AdamW>,
    muon_optimizer: Option<MuonOptimizer>,
    ema: Option<ModelEma>,
}

impl RealTrainer {
    pub fn new(
        config: RealTrainingConfig,
        data_loader: BpeDataLoader,
        tokenizer: BpeTokenizer,
    ) -> Result<Self> {
        let (model, varmap) = create_gpt_model(config.model_config.clone(), &config.device)?;

        let vars: Vec<(String, candle_core::Var)> = {
            let data = varmap.data().lock().unwrap();
            data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        };
        println!("Number of trainable variables: {}", vars.len());

        let (optimizer, muon_optimizer) = if config.use_muon {
            // Split into Muon (2D non-embedding) and AdamW (1D + embeddings) groups
            let (muon_vars, adamw_vars) = split_param_groups(&vars);
            println!("Muon params: {}, AdamW params: {}", muon_vars.len(), adamw_vars.len());

            let muon_opt = MuonOptimizer::new(
                muon_vars,
                adamw_vars,
                config.learning_rate,
                config.muon_momentum,
                config.weight_decay,
                config.learning_rate / 40.0, // adamw_lr = muon_lr / 40
                config.beta1,
                config.beta2,
                config.eps,
                &config.device,
            )?;

            (None, Some(muon_opt))
        } else {
            let optimizer_params = ParamsAdamW {
                lr: config.learning_rate,
                weight_decay: config.weight_decay,
                beta1: config.beta1,
                beta2: config.beta2,
                eps: config.eps,
            };
            let opt_vars: Vec<candle_core::Var> = vars.iter().map(|(_, v)| v.clone()).collect();
            let optimizer = AdamW::new(opt_vars, optimizer_params)?;
            (Some(optimizer), None)
        };

        let ema = if config.use_ema {
            Some(ModelEma::new(&vars, config.ema_target_decay, &config.device)?)
        } else {
            None
        };

        Ok(Self {
            config,
            model,
            varmap,
            data_loader,
            tokenizer,
            optimizer,
            muon_optimizer,
            ema,
        })
    }

    pub fn train(&mut self) -> anyhow::Result<RealTrainingResult> {
        let start_time = Instant::now();
        let mut best_val_loss = f64::INFINITY;
        let mut best_val_loss_ema = f64::INFINITY;
        let mut loss_curve = Vec::new();
        let mut val_loss_curve = Vec::new();

        println!("Starting training for {} steps (EMA: {})", 
            self.config.max_steps, self.config.use_ema);
        let mut progress = ProgressBar::new(self.config.max_steps);

        for step in 0..self.config.max_steps {
            let lr = self.config.lr_scheduler.get_lr(step) as f64;

            if let Some(opt) = &mut self.optimizer {
                opt.set_learning_rate(lr);
            }
            if let Some(opt) = &mut self.muon_optimizer {
                opt.set_learning_rate(lr);
            }

            // Training step
            let train_loss = self.training_step()?;
            loss_curve.push(train_loss);

            // Update EMA after weight update
            if let Some(ema) = &mut self.ema {
                let vars: Vec<(String, candle_core::Var)> = {
                    let data = self.varmap.data().lock().unwrap();
                    data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                };
                ema.update(&vars, step)?;
            }

            // Evaluation
            if step == 0 || step % self.config.eval_interval == 0 || step == self.config.max_steps - 1 {
                // Swap in EMA weights for validation
                if let Some(ema) = &mut self.ema {
                    let mut vars: Vec<(String, candle_core::Var)> = {
                        let data = self.varmap.data().lock().unwrap();
                        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                    };
                    ema.swap_in(&mut vars)?;
                }

                let val_loss = self.evaluate()?;
                val_loss_curve.push(val_loss);

                if val_loss < best_val_loss_ema {
                    best_val_loss_ema = val_loss;
                    println!("New best EMA val loss: {:.6} at step {}", best_val_loss_ema, step);
                }

                // Also compute raw-weight val loss for comparison
                if let Some(ema) = &mut self.ema {
                    let mut vars: Vec<(String, candle_core::Var)> = {
                        let data = self.varmap.data().lock().unwrap();
                        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                    };
                    ema.swap_out(&mut vars)?;
                    let raw_val_loss = self.evaluate()?;
                    if raw_val_loss < best_val_loss {
                        best_val_loss = raw_val_loss;
                    }
                    println!("Step {}: train={:.4}, val_ema={:.4}, val_raw={:.4}, lr={:.5}",
                        step, train_loss, val_loss, raw_val_loss, lr);
                } else {
                    if val_loss < best_val_loss {
                        best_val_loss = val_loss;
                        best_val_loss_ema = val_loss;
                    }
                    println!("Step {}: train={:.4}, val={:.4}, lr={:.5}",
                        step, train_loss, val_loss, lr);
                }
            }

            progress.update(step + 1);
        }

        let training_time = start_time.elapsed();

        Ok(RealTrainingResult {
            best_val_loss,
            best_val_loss_ema,
            steps_completed: self.config.max_steps,
            training_time,
            loss_curve,
            val_loss_curve,
        })
    }

    fn training_step(&mut self) -> Result<f64> {
        let (input, targets) = self.data_loader.get_train_batch()?;
        let logits = self.model.forward(&input)?;
        let loss = self.cross_entropy_loss(&logits, &targets)?;

        if let Some(opt) = &mut self.muon_optimizer {
            opt.backward_step(&loss)?;
        } else if let Some(opt) = &mut self.optimizer {
            opt.backward_step(&loss)?;
        }

        Ok(loss.to_vec0::<f32>()? as f64)
    }

    fn evaluate(&mut self) -> Result<f64> {
        self.data_loader.reset_val_ptr();
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for _ in 0..5 {
            let (input, targets) = self.data_loader.get_val_batch()?;
            let logits = self.model.forward(&input)?;
            let loss = self.cross_entropy_loss(&logits, &targets)?;
            total_loss += loss.to_vec0::<f32>()? as f64;
            num_batches += 1;
        }

        Ok(total_loss / num_batches as f64)
    }

    fn cross_entropy_loss(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, vocab_size) = logits.dims3()?;
        let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
        let targets_flat = targets.reshape((batch_size * seq_len,))?;
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
        Ok(loss)
    }

    pub fn generate_sample(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let mut tokens = self.tokenizer.encode(prompt)?;
        let eos_id = self.tokenizer.eos_id();

        for _ in 0..max_tokens {
            let input = Tensor::from_vec(
                tokens.clone(),
                (1, tokens.len()),
                &self.config.device,
            )?;

            let logits = self.model.forward(&input)?;
            let seq_len = logits.dims()[1];
            let next_token_logits = logits.i((0, seq_len - 1))?;
            let probs = candle_nn::ops::softmax(&next_token_logits, 0)?;
            let next_token = self.sample_from_distribution(&probs)?;

            tokens.push(next_token);

            if next_token == eos_id {
                break;
            }
        }

        self.tokenizer.decode(&tokens)
    }

    fn sample_from_distribution(&self, probs: &Tensor) -> Result<u32> {
        let probs_vec = probs.to_vec1::<f32>()?;
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
