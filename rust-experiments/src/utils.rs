//! Utility functions and structures for experiments

use serde::{Deserialize, Serialize};
use std::time::Duration;


/// Experiment result tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub name: String,
    pub description: String,
    pub best_val_loss: f64,
    pub baseline_loss: f64,
    pub improvement_percent: f64,
    pub training_time: Duration,
    pub steps_completed: usize,
    pub config: serde_json::Value,
    pub status: String,
}

/// Learning rate scheduler trait
pub trait LrScheduler: Send + Sync {
    fn get_lr(&self, step: usize) -> f32;
}

/// Cosine learning rate scheduler
pub struct CosineLrScheduler {
    max_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    max_steps: usize,
}

impl CosineLrScheduler {
    pub fn new(max_lr: f32, min_lr: f32, warmup_steps: usize, max_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr,
            warmup_steps,
            max_steps,
        }
    }
}

impl LrScheduler for CosineLrScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            return self.max_lr * step as f32 / self.warmup_steps as f32;
        }
        
        let progress = (step - self.warmup_steps) as f32 / (self.max_steps - self.warmup_steps) as f32;
        self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())
    }
}

/// Constant learning rate scheduler
pub struct ConstantLrScheduler {
    lr: f32,
}

impl ConstantLrScheduler {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LrScheduler for ConstantLrScheduler {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }
}

/// WSD (Warmup-Stable-Decay) learning rate scheduler.
/// Matches the Python model's `make_wsd_lambda`.
pub struct WsdLrScheduler {
    max_lr: f32,
    warmup_steps: usize,
    decay_start: usize,
    decay_len: usize,
}

impl WsdLrScheduler {
    /// warmup_pct and decay_pct are fractions of max_steps.
    pub fn new(max_lr: f32, max_steps: usize, warmup_pct: f32, decay_pct: f32) -> Self {
        let warmup_steps = std::cmp::max(1, (max_steps as f32 * warmup_pct) as usize);
        let decay_start = max_steps - std::cmp::max(1, (max_steps as f32 * decay_pct) as usize);
        let decay_len = std::cmp::max(1, max_steps - decay_start);
        Self { max_lr, warmup_steps, decay_start, decay_len }
    }
}

impl LrScheduler for WsdLrScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup: (step+1)/warmup_steps
            self.max_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else if step < self.decay_start {
            // Stable
            self.max_lr
        } else {
            // Linear decay to 0
            let p = (step - self.decay_start + 1) as f32 / self.decay_len as f32;
            self.max_lr * (1.0 - p).max(0.0)
        }
    }
}

/// Progress bar for training
pub struct ProgressBar {
    total: usize,
    current: usize,
    start_time: std::time::Instant,
}

impl ProgressBar {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            current: 0,
            start_time: std::time::Instant::now(),
        }
    }
    
    pub fn update(&mut self, current: usize) {
        self.current = current;
        self.print();
    }
    
    pub fn inc(&mut self) {
        self.current += 1;
        self.print();
    }
    
    fn print(&self) {
        let percent = self.current as f64 / self.total as f64;
        let bar_width = 40;
        let filled = (percent * bar_width as f64) as usize;
        let bar = "=".repeat(filled) + &"-".repeat(bar_width - filled);
        
        let elapsed = self.start_time.elapsed();
        let eta = if self.current > 0 {
            Duration::from_secs_f64(elapsed.as_secs_f64() * (self.total - self.current) as f64 / self.current as f64)
        } else {
            Duration::ZERO
        };
        
        print!(
            "\r[{bar}] {percent:.1}% ({}/{}) ETA: {:?}",
            self.current, self.total, eta
        );
        
        if self.current == self.total {
            println!();
        }
        
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_scheduler() {
        let scheduler = CosineLrScheduler::new(0.025, 0.001, 100, 896);
        
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(50), 0.0125);
        assert_eq!(scheduler.get_lr(100), 0.025);
        
        let mid_lr = scheduler.get_lr(498);
        assert!(mid_lr < 0.025 && mid_lr > 0.001);
        
        assert!((scheduler.get_lr(896) - 0.001).abs() < 1e-6);
    }
}