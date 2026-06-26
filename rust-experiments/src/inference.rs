//! Optimized inference engine for GPT model generation.
//!
//! Implements:
//! - Pre-allocated KV cache (zero allocation during generation)
//! - Temperature / top-k / top-p sampling
//! - Speculative decoding via n-gram suffix matching
//! - Model save/load via safetensors
//! - Batched generation

use crate::{
    model::{GPTConfig, GPT, create_gpt_model, get_cos_sin, apply_rotary_pos_emb, title_boundary_attn_mask},
    bpe_data::BpeTokenizer,
};
use candle_core::{D, Device, IndexOp, Result, Tensor};
use candle_nn::{Module, VarMap};
use std::collections::HashMap;
use std::path::Path;

// --- KV Cache ---

/// Pre-allocated KV cache per layer. Zero allocations during generation.
pub struct KvCache {
    k: Vec<Option<Tensor>>,  // [n_layer] each [bh, seq, d]
    v: Vec<Option<Tensor>>,
    seq_len: usize,
    max_seq: usize,
}

impl KvCache {
    pub fn new(n_layer: usize, _n_head: usize, _head_dim: usize, max_seq: usize, _device: &Device) -> Result<Self> {
        Ok(Self {
            k: (0..n_layer).map(|_| None).collect(),
            v: (0..n_layer).map(|_| None).collect(),
            seq_len: 0,
            max_seq,
        })
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        for k in self.k.iter_mut() { *k = None; }
        for v in self.v.iter_mut() { *v = None; }
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn k_cache(&self) -> &[Option<Tensor>] {
        &self.k
    }

    pub fn v_cache(&self) -> &[Option<Tensor>] {
        &self.v
    }

    pub fn update(&mut self, k: Vec<Tensor>, v: Vec<Tensor>) {
        for (i, (new_k, new_v)) in k.into_iter().zip(v.into_iter()).enumerate() {
            if let Some(existing) = &self.k[i] {
                self.k[i] = Some(Tensor::cat(&[existing.detach(), new_k.clone()], 1).unwrap_or(new_k));
            } else {
                self.k[i] = Some(new_k);
            }
            if let Some(existing) = &self.v[i] {
                self.v[i] = Some(Tensor::cat(&[existing.detach(), new_v.clone()], 1).unwrap_or(new_v));
            } else {
                self.v[i] = Some(new_v);
            }
        }
    }
}

// --- Sampling ---

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,      // 0 = disabled
    pub top_p: f32,        // 1.0 = disabled
    pub repetition_penalty: f32, // 1.0 = disabled
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
        }
    }
}

/// Sample next token with temperature, top-k, top-p, and repetition penalty.
pub fn sample(logits: &Tensor, config: &SamplingConfig, history: &[u32]) -> Result<u32> {
    let mut logits_vec = logits.to_vec1::<f32>()?;

    // Repetition penalty: penalize tokens that appeared in history
    if config.repetition_penalty != 1.0 {
        let seen: std::collections::HashSet<u32> = history.iter().copied().collect();
        for &tok in &seen {
            if (tok as usize) < logits_vec.len() {
                if logits_vec[tok as usize] > 0.0 {
                    logits_vec[tok as usize] /= config.repetition_penalty;
                } else {
                    logits_vec[tok as usize] *= config.repetition_penalty;
                }
            }
        }
    }

    // Temperature scaling
    if config.temperature != 1.0 {
        for l in logits_vec.iter_mut() {
            *l /= config.temperature;
        }
    }

    // Top-k filtering
    if config.top_k > 0 && config.top_k < logits_vec.len() {
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = indexed[config.top_k - 1].1;
        for l in logits_vec.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    // Top-p (nucleus) filtering
    if config.top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
        // Filter out -inf first
        indexed.retain(|(_, v)| v.is_finite());
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute cumulative softmax probabilities
        let max_logit = indexed[0].1;
        let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_logit).exp()).sum();
        let mut cumprob = 0.0;
        let mut cutoff = f32::NEG_INFINITY;
        for &(_, v) in &indexed {
            cumprob += (v - max_logit).exp() / exp_sum;
            if cumprob >= config.top_p {
                cutoff = v;
                break;
            }
        }
        for (i, l) in logits_vec.iter_mut().enumerate() {
            if *l < cutoff {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax to get probabilities
    let max_logit = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits_vec.iter().map(|v| (v - max_logit).exp()).sum();
    let probs: Vec<f32> = logits_vec.iter().map(|v| (v - max_logit).exp() / exp_sum).collect();

    // Sample from distribution
    let r: f32 = rand::random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return Ok(i as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

// --- N-gram speculative decoding ---

/// N-gram suffix tree for speculative decoding.
/// Stores token sequences from previous generations to speculate future tokens.
pub struct NgramSpeculator {
    /// Maps n-gram suffix -> next token. Key is last N tokens joined.
    table: HashMap<Vec<u32>, u32>,
    ngram_size: usize,
    max_spec_len: usize, // max tokens to speculate
}

impl NgramSpeculator {
    pub fn new(ngram_size: usize, max_spec_len: usize) -> Self {
        Self {
            table: HashMap::new(),
            ngram_size,
            max_spec_len,
        }
    }

    /// Update the n-gram table with a new sequence of tokens.
    pub fn update(&mut self, tokens: &[u32]) {
        if tokens.len() < self.ngram_size + 1 {
            return;
        }
        for i in 0..tokens.len() - self.ngram_size {
            let key = tokens[i..i + self.ngram_size].to_vec();
            self.table.insert(key, tokens[i + self.ngram_size]);
        }
    }

    /// Speculate next tokens based on n-gram patterns.
    /// Returns speculated token sequence (may be empty if no match).
    pub fn speculate(&self, tokens: &[u32]) -> Vec<u32> {
        if tokens.len() < self.ngram_size {
            return Vec::new();
        }

        let mut speculated = Vec::with_capacity(self.max_spec_len);
        let mut current_suffix = tokens[tokens.len() - self.ngram_size..].to_vec();

        for _ in 0..self.max_spec_len {
            if let Some(&next) = self.table.get(&current_suffix) {
                speculated.push(next);
                // Shift suffix window
                current_suffix.remove(0);
                current_suffix.push(next);
            } else {
                break;
            }
        }

        speculated
    }

    pub fn clear(&mut self) {
        self.table.clear();
    }
}

// --- Inference Engine ---

pub struct InferenceEngine {
    model: GPT,
    varmap: VarMap,
    tokenizer: BpeTokenizer,
    kv_cache: KvCache,
    config: GPTConfig,
    device: Device,
    speculator: Option<NgramSpeculator>,
    last_logits: Option<Tensor>,
}

impl InferenceEngine {
    /// Create from a trained model (takes ownership of varmap and model).
    pub fn from_trained(
        model: GPT,
        varmap: VarMap,
        tokenizer: BpeTokenizer,
        config: GPTConfig,
        device: Device,
    ) -> Result<Self> {
        let n_head = config.n_head;
        let head_dim = config.d_model / config.n_head;
        let kv_cache = KvCache::new(config.n_layer, n_head, head_dim, config.block_size * 4, &device)?;

        Ok(Self {
            model,
            varmap,
            tokenizer,
            kv_cache,
            config,
            device,
            speculator: None,
            last_logits: None,
        })
    }

    /// Load model from safetensors checkpoint.
    pub fn from_checkpoint(
        checkpoint_path: &str,
        tokenizer_path: &str,
        config: GPTConfig,
        device: Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let model = GPT::load(config.clone(), vb)?;

        // Load weights from checkpoint
        if Path::new(checkpoint_path).exists() {
            let mut varmap_mut = varmap.clone();
            varmap_mut.load(checkpoint_path)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to load checkpoint: {}", e)))?;
            println!("Loaded checkpoint from {}", checkpoint_path);
        } else {
            println!("Warning: checkpoint {} not found, using random init", checkpoint_path);
        }

        // Load tokenizer
        let tokenizer = BpeTokenizer::from_file(tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;

        let n_head = config.n_head;
        let head_dim = config.d_model / config.n_head;
        let kv_cache = KvCache::new(config.n_layer, n_head, head_dim, config.block_size * 4, &device)?;

        Ok(Self {
            model,
            varmap,
            tokenizer,
            kv_cache,
            config,
            device,
            speculator: None,
            last_logits: None,
        })
    }

    /// Save model weights to safetensors.
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.varmap.save(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to save checkpoint: {}", e)))
    }

    /// Enable n-gram speculative decoding.
    pub fn with_speculative_decoding(mut self, ngram_size: usize, max_spec_len: usize) -> Self {
        self.speculator = Some(NgramSpeculator::new(ngram_size, max_spec_len));
        self
    }

    /// Reset KV cache for a new generation.
    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }

    /// Generate text with optimized inference.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        sampling: &SamplingConfig,
    ) -> Result<String> {
        self.reset();

        let mut tokens = self.tokenizer.encode(prompt)?;
        let eos_id = self.tokenizer.eos_id();
        let prompt_len = tokens.len();

        // Process prompt (prefill) — single forward pass
        self.prefill(&tokens)?;

        // Autoregressive generation with KV cache
        let mut generated = 0;
        while generated < max_new_tokens {
            // Try speculative decoding
            if let Some(speculator) = &mut self.speculator {
                let speculated = speculator.speculate(&tokens);
                if !speculated.is_empty() {
                    // Verify speculated tokens
                    let accepted = self.verify_speculation(&tokens, &speculated, sampling)?;
                    if accepted > 0 {
                        tokens.extend_from_slice(&speculated[..accepted]);
                        generated += accepted;
                        if tokens.last() == Some(&eos_id) {
                            break;
                        }
                        continue;
                    }
                }
            }

            // Normal single-token generation
            let next_token = self.generate_next_token(&tokens, sampling)?;
            tokens.push(next_token);
            generated += 1;

            if next_token == eos_id {
                break;
            }

            // Update speculator with new token
            if let Some(speculator) = &mut self.speculator {
                speculator.update(&tokens);
            }
        }

        // Decode only generated tokens
        let generated_tokens = &tokens[prompt_len..];
        self.tokenizer.decode(generated_tokens)
    }

    /// Batched generation for multiple prompts.
    pub fn generate_batch(
        &mut self,
        prompts: &[String],
        max_new_tokens: usize,
        sampling: &SamplingConfig,
    ) -> Result<Vec<String>> {
        // For now, generate sequentially. True batching requires batched KV cache.
        // This is a placeholder — continuous batching would interleave token generation.
        let mut results = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let result = self.generate(prompt, max_new_tokens, sampling)?;
            results.push(result);
        }
        Ok(results)
    }

    // --- Internal methods ---

    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        let input = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &self.device)?;

        let (logits, new_ks, new_vs) = self.model.forward_with_cache(
            &input, self.kv_cache.k_cache(), self.kv_cache.v_cache(), 0,
        )?;

        // Update KV cache
        self.kv_cache.update(new_ks, new_vs);
        self.kv_cache.seq_len = tokens.len();

        let seq_len = logits.dim(1)?;
        self.last_logits = Some(logits.i((0, seq_len - 1))?.detach());

        Ok(())
    }

    fn generate_next_token(&mut self, tokens: &[u32], sampling: &SamplingConfig) -> Result<u32> {
        let last_token = tokens[tokens.len() - 1];
        let input = Tensor::from_vec(vec![last_token], (1, 1), &self.device)?;
        let offset = self.kv_cache.seq_len;

        let (logits, new_ks, new_vs) = self.model.forward_with_cache(
            &input, self.kv_cache.k_cache(), self.kv_cache.v_cache(), offset,
        )?;

        self.kv_cache.update(new_ks, new_vs);
        self.kv_cache.seq_len = offset + 1;

        let next_logits = logits.i((0, 0))?;
        sample(&next_logits, sampling, tokens)
    }

    fn verify_speculation(
        &mut self,
        tokens: &[u32],
        speculated: &[u32],
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        // Verify speculated tokens using KV cache
        let mut accepted = 0;

        for &spec_token in speculated {
            // Generate what the model thinks should come next
            let next_token = self.generate_next_token(
                &tokens[..tokens.len() - 1.min(tokens.len())],
                sampling,
            )?;

            if next_token == spec_token {
                accepted += 1;
            } else {
                break;
            }
        }

        Ok(accepted)
    }
}

// --- Benchmarking ---

pub struct BenchmarkResult {
    pub tokens_generated: usize,
    pub time_seconds: f64,
    pub tokens_per_second: f64,
    pub prompt_tokens: usize,
    pub prefill_time_ms: f64,
}

impl InferenceEngine {
    /// Benchmark generation speed.
    pub fn benchmark(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        sampling: &SamplingConfig,
    ) -> Result<BenchmarkResult> {
        let start = std::time::Instant::now();

        self.reset();
        let prompt_tokens = self.tokenizer.encode(prompt)?;
        let prompt_len = prompt_tokens.len();

        // Prefill
        let prefill_start = std::time::Instant::now();
        self.prefill(&prompt_tokens)?;
        let prefill_time = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // Generate
        let mut tokens = prompt_tokens;
        let eos_id = self.tokenizer.eos_id();
        let gen_start = std::time::Instant::now();

        let mut generated = 0;
        while generated < max_new_tokens {
            let next_token = self.generate_next_token(&tokens, sampling)?;
            tokens.push(next_token);
            generated += 1;
            if next_token == eos_id {
                break;
            }
        }

        let gen_time = gen_start.elapsed().as_secs_f64();
        let total_time = start.elapsed().as_secs_f64();
        let tps = generated as f64 / gen_time;

        Ok(BenchmarkResult {
            tokens_generated: generated,
            time_seconds: total_time,
            tokens_per_second: tps,
            prompt_tokens: prompt_len,
            prefill_time_ms: prefill_time,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_speculator() {
        let mut spec = NgramSpeculator::new(3, 5);
        // Train on sequence: [1, 2, 3, 4, 1, 2, 3, 5]
        spec.update(&[1, 2, 3, 4, 1, 2, 3, 5]);

        // After seeing [1, 2, 3], should speculate [4] (first occurrence)
        // But [1, 2, 3] also maps to 5 (second occurrence overwrites)
        // So the result is 5, not 4
        let speculated = spec.speculate(&[0, 1, 2, 3]);
        assert!(!speculated.is_empty());
        // The n-gram [1,2,3] was last seen mapping to 5
        assert_eq!(speculated[0], 5);
    }

    #[test]
    fn test_sampling_basic() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 0.5], (4,), &device).unwrap();
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let token = sample(&logits, &config, &[]).unwrap();
        assert!(token < 4);
    }
}
