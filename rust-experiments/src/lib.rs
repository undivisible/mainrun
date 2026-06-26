//! Rust-based experiments for GPT model optimization

pub mod cosine_lr;
pub mod adaptive_dropout;
pub mod model;
pub mod training;
pub mod data;
pub mod utils;
pub mod simple_data;
pub mod bpe_data;
pub mod ema;
pub mod experiments;
pub mod inference;
pub mod real_training;
pub mod real_cosine_lr;

pub use cosine_lr::CosineLRExperiment;
pub use adaptive_dropout::AdaptiveDropoutExperiment;
pub use model::{GPTConfig, GPT};
pub use training::{Trainer, TrainingConfig};
pub use data::{DataLoader, Tokenizer};
pub use utils::{ExperimentResult, LrScheduler};
pub use cosine_lr::CosineLRConfig;
pub use simple_data::{SimpleDataLoader, SimpleTokenizer};
pub use real_training::{RealTrainer, RealTrainingConfig, RealTrainingResult};
pub use real_cosine_lr::{RealCosineLRExperiment, RealCosineLRConfig};

use candle_core::Device;

/// Get the best available device for training
pub fn get_device() -> Device {
    Device::new_metal(0).unwrap_or(Device::Cpu)
}

/// Run all experiments in sequence
pub async fn run_all_experiments() -> anyhow::Result<Vec<ExperimentResult>> {
    let mut results = Vec::new();
    let device = get_device();
    
    // Experiment 1: REAL Cosine LR scheduling
    println!("🔄 Running REAL cosine LR experiment...");
    let cosine_exp = RealCosineLRExperiment::new(device.clone());
    let cosine_result = tokio::task::block_in_place(|| {
        cosine_exp.run()
    })?;
    results.push(cosine_result);
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_selection() {
        let device = get_device();
        println!("Using device: {:?}", device);
    }
}