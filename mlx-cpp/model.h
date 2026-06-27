#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
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

    mlx::core::array forward_cached(const mlx::core::array& idx,
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
        Block(mlx::core::array a, mlx::core::array b, mlx::core::array c,
              mlx::core::array d, mlx::core::array e, mlx::core::array f,
              mlx::core::array g, mlx::core::array h, mlx::core::array i)
            : attn_norm_w(std::move(a)), qkv_w(std::move(b)), q_norm_w(std::move(c)),
              k_norm_w(std::move(d)), proj_w(std::move(e)), mlp_norm_w(std::move(f)),
              w_gate(std::move(g)), w_up(std::move(h)), w_out(std::move(i)) {}
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

    static std::vector<Block> make_blocks(std::mt19937& gen, const GPTConfig& cfg, int hd, int hidden);
    void init_rope_cache();

    mlx::core::array rmsnorm(const mlx::core::array& x, const mlx::core::array& w);
    mlx::core::array rmsnorm_fast(const mlx::core::array& x, const mlx::core::array& w);
    mlx::core::array dropout_(const mlx::core::array& x);
    mlx::core::array apply_rope(const mlx::core::array& x, const mlx::core::array& cos, const mlx::core::array& sin);
    mlx::core::array make_mask(const mlx::core::array& idx);

    // Quantized matmul helper: x @ w.T using quantized weights
    mlx::core::array qmatmul(const mlx::core::array& x, const QuantW& qw);
    static QuantW quantize_weight(const mlx::core::array& w);
};
