//! Exponential Moving Average of model weights for validation.
//! Uses raw f32 arithmetic to avoid building autograd graph nodes.

use candle_core::{Device, Result, Tensor, Var};
use std::collections::HashMap;

pub struct ModelEma {
    target_decay: f64,
    shadow: HashMap<String, Vec<f32>>,
    backup: Option<HashMap<String, Tensor>>,
    device: Device,
}

impl ModelEma {
    pub fn new(vars: &[(String, Var)], target_decay: f64, device: &Device) -> Result<Self> {
        let mut shadow = HashMap::new();
        for (name, var) in vars {
            let t = var.as_tensor();
            let flat = t.flatten_all()?;
            let data = flat.to_vec1::<f32>()?;
            shadow.insert(name.clone(), data);
        }
        Ok(Self {
            target_decay,
            shadow,
            backup: None,
            device: device.clone(),
        })
    }

    fn decay(&self, step: usize) -> f64 {
        let s = step as f64;
        self.target_decay.min((1.0 + s) / (10.0 + s))
    }

    /// Update shadow weights. Reads current var values as raw f32, does math in Rust.
    pub fn update(&mut self, vars: &[(String, Var)], step: usize) -> Result<()> {
        let d = self.decay(step) as f32;
        let inv_d = 1.0 - d;
        for (name, var) in vars {
            if let Some(shadow) = self.shadow.get_mut(name) {
                let t = var.as_tensor();
                let flat = t.flatten_all()?;
                let current = flat.to_vec1::<f32>()?;
                // EMA: shadow = shadow * d + current * (1 - d)
                for (s, c) in shadow.iter_mut().zip(current.iter()) {
                    *s = *s * d + *c * inv_d;
                }
            }
        }
        Ok(())
    }

    /// Swap EMA weights into model vars. Backs up current values.
    pub fn swap_in(&mut self, vars: &mut [(String, Var)]) -> Result<()> {
        let mut backup = HashMap::new();
        for (name, var) in vars.iter() {
            backup.insert(name.clone(), var.as_tensor().copy()?);
        }
        for (name, var) in vars.iter_mut() {
            if let Some(shadow_data) = self.shadow.get(name) {
                let shape = var.as_tensor().dims().to_vec();
                let tensor = Tensor::from_vec(shadow_data.clone(), shape, &self.device)?;
                var.set(&tensor)?;
            }
        }
        self.backup = Some(backup);
        Ok(())
    }

    /// Restore training weights from backup.
    pub fn swap_out(&mut self, vars: &mut [(String, Var)]) -> Result<()> {
        if let Some(backup) = self.backup.take() {
            for (name, var) in vars.iter_mut() {
                if let Some(saved) = backup.get(name) {
                    var.set(saved)?;
                }
            }
        }
        Ok(())
    }
}
