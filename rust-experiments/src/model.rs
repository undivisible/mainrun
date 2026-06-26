//! GPT model matching the Python implementation: RoPE, causal+title-boundary mask,
//! QK norm, weight tying, dropout, proper init, SwiGLU, NeoX parallel blocks.

use candle_core::{Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_model: usize,
    pub dropout: f32,
    pub eos_id: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub swiglu_mult: f32,
    pub qk_norm: bool,
    pub label_smoothing: f32,
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            vocab_size: 24000,
            block_size: 256,
            n_layer: 8,
            n_head: 9,
            d_model: 576,
            dropout: 0.15,
            eos_id: 1,
            rope_theta: 1000.0,
            rms_eps: 1e-5,
            swiglu_mult: 8.0 / 3.0,
            qk_norm: true,
            label_smoothing: 0.0,
        }
    }
}

// --- RoPE ---

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let half = d / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let neg_x2 = (x2 * -1.0)?;
    Tensor::cat(&[&neg_x2, &x1], D::Minus1)
}

pub fn apply_rotary_pos_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: [bh, t, d], cos/sin: [t, d]
    let cos = cos.unsqueeze(0)?; // [1, t, d]
    let sin = sin.unsqueeze(0)?;
    let x_cos = x.broadcast_mul(&cos)?;
    let x_rot = rotate_half(x)?;
    let x_sin = x_rot.broadcast_mul(&sin)?;
    x_cos + x_sin
}

pub fn get_cos_sin(seq_len: usize, head_dim: usize, theta: f32, device: &candle_core::Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (half,), device)?;
    let t = Tensor::arange(0f32, seq_len as f32, device)?.reshape((seq_len, 1))?;
    let freqs = t.matmul(&inv_freq.reshape((1, half))?)?; // [t, half]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [t, d]
    let cos = emb.cos()?;
    let sin = emb.sin()?;
    Ok((cos, sin))
}

// --- Title boundary + causal attention mask ---

pub fn title_boundary_attn_mask(idx: &Tensor, eos_id: usize) -> Result<Tensor> {
    let (_b, t) = idx.dims2()?;
    let device = idx.device();

    // Build segment IDs: cumsum of eos positions minus eos itself
    let eos = idx.eq(eos_id as i64)?;
    let eos_f = eos.to_dtype(candle_core::DType::F32)?;
    let cumsum = eos_f.cumsum(D::Minus1)?;
    let seg = (cumsum - &eos_f)?;

    // same_seg[b, i, j] = 1 if seg[i] == seg[j]
    // seg: [b, t], need [b, t, t]
    let seg_i = seg.unsqueeze(2)?; // [b, t, 1]
    let seg_j = seg.unsqueeze(1)?; // [b, 1, t]
    // eq doesn't broadcast, use subtraction + threshold
    let diff = seg_i.broadcast_sub(&seg_j)?;
    let same_seg = diff.eq(&Tensor::new(0f32, device)?.broadcast_as(diff.dims())?)?;
    let same_seg = same_seg.to_dtype(candle_core::DType::F32)?; // [b, t, t]

    // causal mask: tril
    let causal = Tensor::tril2(t, candle_core::DType::F32, device)?;

    // allowed = causal & same_seg (1.0 where allowed, 0.0 where not)
    let allowed = causal.broadcast_mul(&same_seg)?;

    // mask: 0 where allowed, large negative where not
    // Use -1e9 instead of -inf to avoid NaN from 0 * -inf
    let ones = Tensor::ones_like(&allowed)?;
    let neg_allowed = ones.broadcast_sub(&allowed)?;
    let neg_val = Tensor::new(-1e9f32, device)?;
    let mask = neg_allowed.broadcast_mul(&neg_val)?;
    // mask is [b, t, t], we need [b, 1, t, t] for attention
    mask.unsqueeze(1)
}

// --- RMSNorm wrapper (candle_nn has RmsNorm but we need weight access) ---

// --- Attention ---

#[derive(Debug)]
pub struct CausalSelfAttention {
    n_head: usize,
    head_dim: usize,
    qkv: Linear,
    proj: Linear,
    q_norm: Option<candle_nn::RmsNorm>,
    k_norm: Option<candle_nn::RmsNorm>,
    dropout_p: f32,
    rope_theta: f32,
}

impl CausalSelfAttention {
    pub fn new(config: &GPTConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.d_model / config.n_head;
        let qkv = candle_nn::linear_no_bias(config.d_model, 3 * config.d_model, vb.pp("qkv"))?;
        let proj = candle_nn::linear_no_bias(config.d_model, config.d_model, vb.pp("proj"))?;

        let q_norm = if config.qk_norm {
            Some(candle_nn::rms_norm(head_dim, config.rms_eps.into(), vb.pp("q_norm"))?)
        } else {
            None
        };
        let k_norm = if config.qk_norm {
            Some(candle_nn::rms_norm(head_dim, config.rms_eps.into(), vb.pp("k_norm"))?)
        } else {
            None
        };

        Ok(Self {
            n_head: config.n_head,
            head_dim,
            qkv,
            proj,
            q_norm,
            k_norm,
            dropout_p: config.dropout,
            rope_theta: config.rope_theta,
        })
    }

    pub fn forward(&self, x: &Tensor, attn_mask: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, attn_mask, None, None, 0)
    }

    /// Forward with KV cache for incremental decoding.
    /// `cache_k`/`cache_v` are the cached keys/values from previous tokens.
    /// `offset` is the number of tokens already processed (for RoPE position).
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        attn_mask: &Tensor,
        cache_k: Option<&Tensor>,
        cache_v: Option<&Tensor>,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // We need to return new k, v for cache update
        // Reuse forward_inner but also return k, v
        self.forward_inner_return_kv(x, attn_mask, cache_k, cache_v, offset)
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        attn_mask: &Tensor,
        cache_k: Option<&Tensor>,
        cache_v: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let device = x.device();

        let qkv = self.qkv.forward(x)?;
        let chunks = qkv.chunk(3, D::Minus1)?;
        let q = &chunks[0];
        let k = &chunks[1];
        let v = &chunks[2];

        let q = q.reshape((b, t, self.n_head, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .reshape((b * self.n_head, t, self.head_dim))?;
        let k = k.reshape((b, t, self.n_head, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .reshape((b * self.n_head, t, self.head_dim))?;
        let v = v.reshape((b, t, self.n_head, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .reshape((b * self.n_head, t, self.head_dim))?;

        // RoPE — use offset for position encoding during incremental decoding
        let total_len = offset + t;
        let (cos, sin) = get_cos_sin(total_len, self.head_dim, self.rope_theta, device)?;
        let cos = cos.narrow(0, offset, t)?;
        let sin = sin.narrow(0, offset, t)?;
        let q = apply_rotary_pos_emb(&q, &cos, &sin)?;
        let k = apply_rotary_pos_emb(&k, &cos, &sin)?;

        let q = if let Some(qn) = &self.q_norm { qn.forward(&q)? } else { q };
        let k = if let Some(kn) = &self.k_norm { kn.forward(&k)? } else { k };

        // Append to KV cache
        let (k_full, v_full) = if let (Some(ck), Some(cv)) = (cache_k, cache_v) {
            let k_cat = Tensor::cat(&[ck.detach(), k.clone()], 1)?;
            let v_cat = Tensor::cat(&[cv.detach(), v.clone()], 1)?;
            (k_cat, v_cat)
        } else {
            (k, v)
        };

        let kv_len = k_full.dim(1)?;

        // Attention: q [bh, t, d] @ k_full [bh, kv_len, d].T -> [bh, t, kv_len]
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scale_tensor = Tensor::new(scale, device)?;
        let attn = q.matmul(&k_full.t()?)?;
        let attn = attn.broadcast_mul(&scale_tensor)?;

        // Mask: during incremental decode with t=1, mask is just [b, 1, 1, kv_len]
        // The last row of the causal mask (allow all up to position offset+t)
        let mask_flat = if t == 1 {
            // Single token: attend to all cached tokens (no masking needed beyond causal)
            // Causal is automatically satisfied since offset >= 0
            attn_mask
                .broadcast_as((b, self.n_head, 1, kv_len))?
                .reshape((b * self.n_head, 1, kv_len))?
        } else {
            attn_mask
                .broadcast_as((b, self.n_head, t, kv_len))?
                .reshape((b * self.n_head, t, kv_len))?
        };
        let attn = attn.broadcast_add(&mask_flat)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v_full)?;

        let out = out.reshape((b, self.n_head, t, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .reshape((b, t, c))?;
        self.proj.forward(&out)
    }

    fn forward_inner_return_kv(
        &self,
        x: &Tensor,
        attn_mask: &Tensor,
        cache_k: Option<&Tensor>,
        cache_v: Option<&Tensor>,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, t, c) = x.dims3()?;
        let device = x.device();

        let qkv = self.qkv.forward(x)?;
        let chunks = qkv.chunk(3, D::Minus1)?;
        let q = &chunks[0];
        let k = &chunks[1];
        let v = &chunks[2];

        let q = q.reshape((b, t, self.n_head, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .reshape((b * self.n_head, t, self.head_dim))?;
        let k = k.reshape((b, t, self.n_head, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .reshape((b * self.n_head, t, self.head_dim))?;
        let v = v.reshape((b, t, self.n_head, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .reshape((b * self.n_head, t, self.head_dim))?;

        let total_len = offset + t;
        let (cos, sin) = get_cos_sin(total_len, self.head_dim, self.rope_theta, device)?;
        let cos = cos.narrow(0, offset, t)?;
        let sin = sin.narrow(0, offset, t)?;
        let q = apply_rotary_pos_emb(&q, &cos, &sin)?;
        let k_new = apply_rotary_pos_emb(&k, &cos, &sin)?;
        let v_new = v;

        let q = if let Some(qn) = &self.q_norm { qn.forward(&q)? } else { q };
        let k_new = if let Some(kn) = &self.k_norm { kn.forward(&k_new)? } else { k_new };

        let (k_full, v_full) = if let (Some(ck), Some(cv)) = (cache_k, cache_v) {
            let k_cat = Tensor::cat(&[ck.detach(), k_new.clone()], 1)?;
            let v_cat = Tensor::cat(&[cv.detach(), v_new.clone()], 1)?;
            (k_cat, v_cat)
        } else {
            (k_new.clone(), v_new.clone())
        };

        let kv_len = k_full.dim(1)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scale_tensor = Tensor::new(scale, device)?;
        let attn = q.matmul(&k_full.t()?)?;
        let attn = attn.broadcast_mul(&scale_tensor)?;

        let mask_flat = if t == 1 {
            attn_mask
                .broadcast_as((b, self.n_head, 1, kv_len))?
                .reshape((b * self.n_head, 1, kv_len))?
        } else {
            attn_mask
                .broadcast_as((b, self.n_head, t, kv_len))?
                .reshape((b * self.n_head, t, kv_len))?
        };
        let attn = attn.broadcast_add(&mask_flat)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v_full)?;
        let out = out.reshape((b, self.n_head, t, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .reshape((b, t, c))?;
        let out = self.proj.forward(&out)?;

        Ok((out, k_new.detach(), v_new.detach()))
    }
}

// --- SwiGLU MLP ---

#[derive(Debug)]
pub struct SwiGLU {
    w_gate: Linear,
    w_up: Linear,
    w_out: Linear,
}

impl SwiGLU {
    pub fn new(config: &GPTConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = (config.swiglu_mult * config.d_model as f32) as usize;
        let w_gate = candle_nn::linear_no_bias(config.d_model, hidden, vb.pp("w_gate"))?;
        let w_up = candle_nn::linear_no_bias(config.d_model, hidden, vb.pp("w_up"))?;
        let w_out = candle_nn::linear_no_bias(hidden, config.d_model, vb.pp("w_out"))?;
        Ok(Self { w_gate, w_up, w_out })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w_gate.forward(x)?)?;
        let up = self.w_up.forward(x)?;
        self.w_out.forward(&(gate * up)?)
    }
}

// --- Block (NeoX parallel) ---

#[derive(Debug)]
pub struct Block {
    attn_norm: candle_nn::RmsNorm,
    mlp_norm: candle_nn::RmsNorm,
    attn: CausalSelfAttention,
    mlp: SwiGLU,
}

impl Block {
    pub fn new(config: &GPTConfig, vb: VarBuilder) -> Result<Self> {
        let attn_norm = candle_nn::rms_norm(config.d_model, config.rms_eps.into(), vb.pp("attn_norm"))?;
        let mlp_norm = candle_nn::rms_norm(config.d_model, config.rms_eps.into(), vb.pp("mlp_norm"))?;
        let attn = CausalSelfAttention::new(config, vb.pp("attn"))?;
        let mlp = SwiGLU::new(config, vb.pp("mlp"))?;
        Ok(Self { attn_norm, mlp_norm, attn, mlp })
    }

    pub fn forward(&self, x: &Tensor, attn_mask: &Tensor) -> Result<Tensor> {
        let attn_out = self.attn.forward(&self.attn_norm.forward(x)?, attn_mask)?;
        let mlp_out = self.mlp.forward(&self.mlp_norm.forward(x)?)?;
        Ok((x + attn_out + mlp_out)?)
    }

    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        attn_mask: &Tensor,
        cache_k: Option<&Tensor>,
        cache_v: Option<&Tensor>,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let normed = self.attn_norm.forward(x)?;
        let (attn_out, new_k, new_v) = self.attn.forward_with_cache(
            &normed, attn_mask, cache_k, cache_v, offset,
        )?;
        let mlp_out = self.mlp.forward(&self.mlp_norm.forward(x)?)?;
        let out = (x + attn_out + mlp_out)?;
        Ok((out, new_k, new_v))
    }
}

// --- GPT ---

#[derive(Debug)]
pub struct GPT {
    config: GPTConfig,
    token_emb: Embedding,
    blocks: Vec<Block>,
    ln_f: candle_nn::RmsNorm,
    // head is tied to token_emb — we store the weight tensor separately
    head_weight: Tensor,
}

impl GPT {
    pub fn load(config: GPTConfig, vb: VarBuilder) -> Result<Self> {
        let token_emb = candle_nn::embedding(config.vocab_size, config.d_model, vb.pp("token_emb"))?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let block = Block::new(&config, vb.pp(&format!("blocks.{}", i)))?;
            blocks.push(block);
        }

        let ln_f = candle_nn::rms_norm(config.d_model, config.rms_eps.into(), vb.pp("ln_f"))?;

        // Tied head: use token_emb weight for projection
        let head_weight = vb.get((config.vocab_size, config.d_model), "token_emb.weight")?;

        Ok(Self {
            config,
            token_emb,
            blocks,
            ln_f,
            head_weight,
        })
    }

    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let (b, t) = idx.dims2()?;
        let attn_mask = title_boundary_attn_mask(idx, self.config.eos_id)?;

        let mut x = self.token_emb.forward(idx)?;

        for block in &self.blocks {
            x = block.forward(&x, &attn_mask)?;
        }

        x = self.ln_f.forward(&x)?;

        let bt = x.dim(0)? * x.dim(1)?;
        let d = x.dim(2)?;
        let x_flat = x.reshape((bt, d))?;
        let head_t = self.head_weight.t()?;
        let logits_flat = x_flat.matmul(&head_t)?;
        let logits = logits_flat.reshape((b, t, self.config.vocab_size))?;
        Ok(logits)
    }

    /// Forward with KV cache for incremental decoding.
    /// Returns logits and per-layer (k, v) tensors for cache storage.
    pub fn forward_with_cache(
        &self,
        idx: &Tensor,
        cache_k: &[Option<Tensor>],
        cache_v: &[Option<Tensor>],
        offset: usize,
    ) -> Result<(Tensor, Vec<Tensor>, Vec<Tensor>)> {
        let (b, t) = idx.dims2()?;
        let attn_mask = title_boundary_attn_mask(idx, self.config.eos_id)?;

        let mut x = self.token_emb.forward(idx)?;
        let mut new_ks = Vec::with_capacity(self.blocks.len());
        let mut new_vs = Vec::with_capacity(self.blocks.len());

        for (i, block) in self.blocks.iter().enumerate() {
            let (out, new_k, new_v) = block.forward_with_cache(
                &x, &attn_mask,
                cache_k[i].as_ref(), cache_v[i].as_ref(),
                offset,
            )?;
            x = out;
            new_ks.push(new_k);
            new_vs.push(new_v);
        }

        x = self.ln_f.forward(&x)?;
        let bt = x.dim(0)? * x.dim(1)?;
        let d = x.dim(2)?;
        let x_flat = x.reshape((bt, d))?;
        let head_t = self.head_weight.t()?;
        let logits_flat = x_flat.matmul(&head_t)?;
        let logits = logits_flat.reshape((b, t, self.config.vocab_size))?;

        Ok((logits, new_ks, new_vs))
    }

    pub fn config(&self) -> &GPTConfig {
        &self.config
    }
}

/// Create a GPT model with trainable parameters using VarMap.
/// Applies proper weight initialization matching the Python model.
pub fn create_gpt_model(config: GPTConfig, device: &candle_core::Device) -> Result<(GPT, VarMap)> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let model = GPT::load(config, vb)?;

    // Initialize weights: normal(0, 0.02) for all parameters
    // Use the rand crate's normal distribution for numerical stability
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(42);
    let std = 0.02f32;
    let n_layer = model.config.n_layer as f32;

    let data = varmap.data().lock().unwrap();
    for (name, var) in data.iter() {
        let shape = var.as_tensor().dims();
        let n_elems: usize = shape.iter().product();

        // Scaled init for proj and w_out: std / sqrt(2 * n_layer)
        let scaled_std = if name.contains("attn.proj.weight") || name.contains("mlp.w_out.weight") {
            std / (2.0 * n_layer).sqrt()
        } else {
            std
        };

        let normal = Normal::<f32>::new(0.0, scaled_std).unwrap();
        let init_data: Vec<f32> = (0..n_elems)
            .map(|_| {
                let v = normal.sample(&mut rng);
                // Clamp to avoid inf/NaN
                if v.is_finite() { v } else { 0.0 }
            })
            .collect();

        let init_tensor = Tensor::from_vec(init_data, shape, device)?;
        var.set(&init_tensor)?;
    }
    drop(data);

    Ok((model, varmap))
}
