//! Muon optimizer: Momentum + Orthogonalization.
//!
//! For 2D weight matrices, Muon applies momentum then orthogonalizes
//! the update via Newton-Schulz iteration (approximating polar decomposition).
//! For 1D parameters (norms, embeddings), falls back to AdamW.

use candle_core::{Device, Result, Tensor, Var};
use candle_core::backprop::GradStore;

pub struct MuonOptimizer {
    muon_vars: Vec<Var>,
    adamw_vars: Vec<Var>,
    muon_momentum: Vec<Tensor>,
    adamw_m: Vec<Tensor>,
    adamw_v: Vec<Tensor>,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    adamw_lr: f64,
    adamw_beta1: f64,
    adamw_beta2: f64,
    adamw_eps: f64,
    step: usize,
    device: Device,
}

/// Multiply a tensor by a scalar f64. Works around Candle's limited operator overloading.
fn scalar_tensor(s: f32, device: &Device) -> Result<Tensor> {
    Tensor::from_vec(vec![s], (1,), device)
}

fn mul_scalar(t: &Tensor, s: f64, device: &Device) -> Result<Tensor> {
    let scalar = scalar_tensor(s as f32, device)?;
    t.broadcast_mul(&scalar)
}

fn div_scalar(t: &Tensor, s: f64, device: &Device) -> Result<Tensor> {
    let scalar = scalar_tensor(s as f32, device)?;
    t.broadcast_div(&scalar)
}

fn add_scalar(t: &Tensor, s: f64, device: &Device) -> Result<Tensor> {
    let scalar = scalar_tensor(s as f32, device)?;
    t.broadcast_add(&scalar)
}

impl MuonOptimizer {
    pub fn new(
        muon_vars: Vec<Var>,
        adamw_vars: Vec<Var>,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
        adamw_lr: f64,
        adamw_beta1: f64,
        adamw_beta2: f64,
        adamw_eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let muon_momentum = muon_vars.iter().map(|v| Tensor::zeros_like(v.as_tensor())).collect::<Result<Vec<_>>>()?;
        let adamw_m = adamw_vars.iter().map(|v| Tensor::zeros_like(v.as_tensor())).collect::<Result<Vec<_>>>()?;
        let adamw_v = adamw_vars.iter().map(|v| Tensor::zeros_like(v.as_tensor())).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            muon_vars, adamw_vars, muon_momentum, adamw_m, adamw_v,
            lr, momentum, weight_decay, adamw_lr,
            adamw_beta1, adamw_beta2, adamw_eps, step: 0,
            device: device.clone(),
        })
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
        self.adamw_lr = lr / 40.0;
    }

    fn newton_schulz5(&self, g: &Tensor, steps: usize) -> Result<Tensor> {
        let (rows, cols) = g.dims2()?;
        let g = if rows < cols { g.t()? } else { g.clone() };

        let frob = g.sqr()?.sum_all()?.sqrt()?.to_vec0::<f32>()?.max(1e-8);
        let mut x = div_scalar(&g, frob as f64, &self.device)?;

        let a = 3.4445f64;
        let b = -4.7750f64;
        let c = 2.0315f64;

        for _ in 0..steps {
            let x_t = x.t()?;
            let a_mat = x_t.matmul(&x)?;
            let a_sq = a_mat.matmul(&a_mat)?;
            let b_mat = mul_scalar(&a_mat, b, &self.device)?;
            let c_mat = mul_scalar(&a_sq, c, &self.device)?;
            let b_total = b_mat.add(&c_mat)?;
            let bx = x.matmul(&b_total)?;
            let ax = mul_scalar(&x, a, &self.device)?;
            x = ax.add(&bx)?;
        }

        if rows < cols { x.t() } else { Ok(x) }
    }

    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step += 1;

        // Muon update for 2D parameters
        for (i, var) in self.muon_vars.iter().enumerate() {
            let grad = match grads.get(var) {
                Some(g) => g,
                None => continue,
            };
            if grad.rank() != 2 { continue; }
            let grad = grad.detach();
            let current = var.as_tensor().detach();

            let wd_scaled = mul_scalar(&current, self.weight_decay, &self.device)?;
            let grad_wd = grad.add(&wd_scaled)?;

            let mom_prev = &self.muon_momentum[i];
            let mom_scaled = mul_scalar(mom_prev, self.momentum, &self.device)?;
            let grad_scaled = mul_scalar(&grad_wd, 1.0 - self.momentum, &self.device)?;
            let new_momentum = mom_scaled.add(&grad_scaled)?;
            self.muon_momentum[i] = new_momentum.clone();

            let orthogonalized = self.newton_schulz5(&new_momentum, 5)?;
            let (r, c) = new_momentum.dims2()?;
            let scale = ((r.max(c) as f64) / (r.min(c) as f64)).sqrt() * self.lr;
            let update = mul_scalar(&orthogonalized, scale, &self.device)?;
            var.set(&current.sub(&update)?)?;
        }

        // AdamW for 1D / embeddings
        let bc1 = 1.0 - self.adamw_beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.adamw_beta2.powi(self.step as i32);

        for (i, var) in self.adamw_vars.iter().enumerate() {
            let grad = match grads.get(var) {
                Some(g) => g,
                None => continue,
            };
            let grad = grad.detach();
            let current = var.as_tensor().detach();

            let wd_scaled = mul_scalar(&current, self.weight_decay, &self.device)?;
            let grad_wd = grad.add(&wd_scaled)?;

            let m_prev = &self.adamw_m[i];
            let v_prev = &self.adamw_v[i];

            let m_scaled = mul_scalar(m_prev, self.adamw_beta1, &self.device)?;
            let grad_m = mul_scalar(&grad_wd, 1.0 - self.adamw_beta1, &self.device)?;
            let new_m = m_scaled.add(&grad_m)?;

            let v_scaled = mul_scalar(v_prev, self.adamw_beta2, &self.device)?;
            let grad_sq = grad_wd.sqr()?;
            let grad_v = mul_scalar(&grad_sq, 1.0 - self.adamw_beta2, &self.device)?;
            let new_v = v_scaled.add(&grad_v)?;

            self.adamw_m[i] = new_m.clone();
            self.adamw_v[i] = new_v.clone();

            let m_hat = div_scalar(&new_m, bc1, &self.device)?;
            let v_hat = div_scalar(&new_v, bc2, &self.device)?;
            let v_sqrt = v_hat.sqrt()?;
            let denom = add_scalar(&v_sqrt, self.adamw_eps, &self.device)?;
            let update = m_hat.broadcast_div(&denom)?;
            let update_scaled = mul_scalar(&update, self.adamw_lr, &self.device)?;
            var.set(&current.sub(&update_scaled)?)?;
        }
        Ok(())
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }
}

pub fn split_param_groups(vars: &[(String, Var)]) -> (Vec<Var>, Vec<Var>) {
    let mut muon_vars = Vec::new();
    let mut adamw_vars = Vec::new();
    for (name, var) in vars {
        let rank = var.as_tensor().rank();
        let is_embedding = name.contains("token_emb") || name.contains("head") || name.contains("embed");
        let is_norm = name.contains("norm") || name.contains("ln_f");
        if rank == 2 && !is_embedding && !is_norm {
            muon_vars.push(var.clone());
        } else {
            adamw_vars.push(var.clone());
        }
    }
    (muon_vars, adamw_vars)
}
