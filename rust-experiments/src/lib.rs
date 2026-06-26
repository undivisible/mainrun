//! GPT model training in Rust using Candle.

pub mod model;
pub mod utils;
pub mod bpe_data;
pub mod ema;
pub mod experiments;
pub mod inference;
pub mod muon;
pub mod real_training;
pub mod real_cosine_lr;

pub use model::{GPTConfig, GPT};
pub use utils::{ExperimentResult, LrScheduler};
pub use real_training::{RealTrainer, RealTrainingConfig, RealTrainingResult};
pub use real_cosine_lr::{RealCosineLRExperiment, RealCosineLRConfig};

use candle_core::Device;

/// Get the best available device for training
pub fn get_device() -> Device {
    Device::new_metal(0).unwrap_or(Device::Cpu)
}
