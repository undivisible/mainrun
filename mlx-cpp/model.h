#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <functional>
#include <random>
#include <utility>
#include <vector>

struct GPTConfig {
    int vocab_size = 24000;
    int block_size = 256;
    int n_layer = 8;
    int n_head = 9;
    int d_model = 576;
    float dropout = 0.15f;
    int eos_id = 1;
    float rope_theta = 1000.0f;
    float rms_eps = 1e-5f;
    float swiglu_mult = 8.0f / 3.0f;
    bool qk_norm = true;
};

class GPT {
public:
    explicit GPT(const GPTConfig& cfg);

    mlx::core::array forward(const mlx::core::array& idx, bool train);
    mlx::core::array loss(const mlx::core::array& idx, const mlx::core::array& targets, bool train);

    mlx::core::array forward_functional(const std::vector<mlx::core::array>& params,
                                        const mlx::core::array& idx, bool train) const;
    mlx::core::array loss_functional(const std::vector<mlx::core::array>& params,
                                     const mlx::core::array& idx,
                                     const mlx::core::array& targets, bool train) const;

    mlx::core::array forward_cached(const mlx::core::array& idx,
                                    mlx::core::array& k_cache,
                                    mlx::core::array& v_cache,
                                    int prev_len);

    // Convert parameters to a smaller dtype for inference (bfloat16 by default).
    void promote_for_inference(mlx::core::Dtype dtype = mlx::core::bfloat16);

    // Decode with a growing per-layer KV cache (used for quantized path).
    mlx::core::array forward_decode_growing(
        const mlx::core::array& idx,
        std::vector<std::pair<mlx::core::array, mlx::core::array>>& kv_cache,
        int prev_len);

    std::vector<mlx::core::array*> parameters();
    void set_parameters(const std::vector<mlx::core::array>& params);

    GPTConfig config() const { return cfg_; }

    const mlx::core::array& rope_cos() const { return rope_cos_; }
    const mlx::core::array& rope_sin() const { return rope_sin_; }

    // Quantize weights for fast inference (4-bit, group=64).
    // Call after set_parameters() with trained weights.
    void quantize_for_inference();

    // Whether weights have been quantized.
    bool is_quantized() const { return quantized_; }

private:
    GPTConfig cfg_;
    int head_dim_;
    int hidden_;

    mlx::core::array token_emb_;

    struct Block {
        mlx::core::array attn_norm_w;
        mlx::core::array qkv_w;
        mlx::core::array q_norm_w;
        mlx::core::array k_norm_w;
        mlx::core::array proj_w;
        mlx::core::array mlp_norm_w;
        mlx::core::array w_gate;
        mlx::core::array w_up;
        mlx::core::array w_out;
    };

    std::vector<Block> blocks_;
    mlx::core::array ln_f_w_;

    mlx::core::array rope_cos_;
    mlx::core::array rope_sin_;

    bool quantized_ = false;
    struct QuantW {
        mlx::core::array w;
        mlx::core::array scales;
        mlx::core::array biases;
        QuantW() : w({0.0f}, mlx::core::float32), scales({0.0f}, mlx::core::float32), biases({0.0f}, mlx::core::float32) {}
        QuantW(mlx::core::array w_, mlx::core::array s_, mlx::core::array b_) : w(std::move(w_)), scales(std::move(s_)), biases(std::move(b_)) {}
    };
    std::vector<QuantW> q_qkv_;
    std::vector<QuantW> q_proj_;
    std::vector<QuantW> q_gate_;
    std::vector<QuantW> q_up_;
    std::vector<QuantW> q_out_;
    QuantW q_emb_;

    mlx::core::Dtype infer_dtype_ = mlx::core::float32;
    mutable std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)> compiled_decode_fp32_;

    static std::vector<Block> make_blocks(std::mt19937& gen, const GPTConfig& cfg, int hd, int hidden);
    void init_rope_cache();

    // One decode step (T=1) with pre-allocated stable KV cache.
    std::vector<mlx::core::array> decode_step_impl(const std::vector<mlx::core::array>& inputs);

    mlx::core::array rmsnorm(const mlx::core::array& x, const mlx::core::array& w) const;
    mlx::core::array dropout_(const mlx::core::array& x) const;
    mlx::core::array make_mask(const mlx::core::array& idx) const;

    // Quantized matmul helper: x @ w.T using quantized weights
    mlx::core::array qmatmul(const mlx::core::array& x, const QuantW& qw) const;
    static QuantW quantize_weight(const mlx::core::array& w);

    // Shared per-layer helpers (reduce duplication across forward variants).
    // Returns {q, k, v} transposed to [B, n_head, T, head_dim] with RoPE + QK-norm applied.
    // If quantized_, uses q_qkv_[layer_idx] for the matmul (ignores qkv_w).
    struct QKV { mlx::core::array q, k, v; };
    QKV compute_qkv(const mlx::core::array& h_flat, int B, int T,
                    const mlx::core::array& qkv_w,
                    const mlx::core::array& q_norm_w,
                    const mlx::core::array& k_norm_w,
                    int layer_idx, int offset) const;
    // Returns mlp_out [B, T, d_model]. If quantized_, uses q_gate_/q_up_/q_out_.
    mlx::core::array compute_mlp(const mlx::core::array& x, int B, int T,
                                 const mlx::core::array& mlp_norm_w,
                                 const mlx::core::array& w_gate,
                                 const mlx::core::array& w_up,
                                 const mlx::core::array& w_out,
                                 int layer_idx) const;
};
