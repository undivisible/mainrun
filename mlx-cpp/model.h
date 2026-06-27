#pragma once

#include <mlx/mlx.h>
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

    std::vector<mlx::core::array*> parameters();
    void set_parameters(const std::vector<mlx::core::array>& params);

    GPTConfig config() const { return cfg_; }

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

    static std::vector<Block> make_blocks(std::mt19937& gen, const GPTConfig& cfg, int hd, int hidden);

    mlx::core::array rmsnorm(const mlx::core::array& x, const mlx::core::array& w);
    mlx::core::array dropout_(const mlx::core::array& x);
    std::pair<mlx::core::array, mlx::core::array> rope_cos_sin(int seq_len);
    mlx::core::array apply_rope(const mlx::core::array& x, const mlx::core::array& cos, const mlx::core::array& sin);
    mlx::core::array make_mask(const mlx::core::array& idx);
};
