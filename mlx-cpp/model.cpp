#include "model.h"

#include <cmath>
#include <random>

using namespace mlx::core;

// ponytail: single helper for N(0, std) weight init with fixed seed
static array normal_array(std::mt19937& gen, float mean, float std, Shape shape) {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    std::vector<float> data(n);
    std::normal_distribution<float> dist(mean, std);
    for (auto& v : data) v = dist(gen);
    return array(data.begin(), std::move(shape), float32);
}

// ponytail: init all params in order with single sequential RNG
std::vector<GPT::Block> GPT::make_blocks(std::mt19937& gen, const GPTConfig& cfg, int hd, int hidden) {
    std::vector<GPT::Block> blocks;
    float std = 0.02f;
    float scaled = std / std::sqrt(2.0f * static_cast<float>(cfg.n_layer));
    for (int i = 0; i < cfg.n_layer; i++) {
        blocks.emplace_back(
            ones({cfg.d_model}),
            normal_array(gen, 0.0f, std, {3 * cfg.d_model, cfg.d_model}),
            ones({hd}),
            ones({hd}),
            normal_array(gen, 0.0f, scaled, {cfg.d_model, cfg.d_model}),
            ones({cfg.d_model}),
            normal_array(gen, 0.0f, std, {hidden, cfg.d_model}),
            normal_array(gen, 0.0f, std, {hidden, cfg.d_model}),
            normal_array(gen, 0.0f, scaled, {cfg.d_model, hidden}));
    }
    return blocks;
}

void GPT::init_rope_cache() {
    int hd = head_dim_;
    int half = hd / 2;
    int T = cfg_.block_size;
    std::vector<float> inv_freq_data(half);
    for (int i = 0; i < half; i++)
        inv_freq_data[i] = 1.0f / std::pow(cfg_.rope_theta, 2.0f * i / hd);
    array inv_freq(inv_freq_data.begin(), {half}, float32);

    auto freqs = outer(arange(0.0, double(T), float32), inv_freq);
    auto emb = concatenate({freqs, freqs}, -1);
    rope_cos_ = reshape(cos(emb), {1, 1, T, hd});
    rope_sin_ = reshape(sin(emb), {1, 1, T, hd});
    eval(rope_cos_, rope_sin_);
}

GPT::GPT(const GPTConfig& cfg)
    : cfg_(cfg),
      head_dim_(cfg.d_model / cfg.n_head),
      hidden_(static_cast<int>(cfg.swiglu_mult * cfg.d_model)),
      token_emb_(0.0f),
      ln_f_w_(0.0f),
      rope_cos_({1, 1, cfg.block_size, cfg.d_model / cfg.n_head}, float32),
      rope_sin_({1, 1, cfg.block_size, cfg.d_model / cfg.n_head}, float32) {
    std::mt19937 gen(1337);
    float std = 0.02f;
    token_emb_ = normal_array(gen, 0.0f, std, {cfg_.vocab_size, cfg_.d_model});
    blocks_ = make_blocks(gen, cfg_, head_dim_, hidden_);
    ln_f_w_ = ones({cfg_.d_model});
    init_rope_cache();
}

array GPT::rmsnorm(const array& x, const array& w) {
    auto ms = mean(square(x), -1, true);
    auto norm = rsqrt(ms + cfg_.rms_eps);
    return w * norm * x;
}

array GPT::dropout_(const array& x) {
    if (cfg_.dropout <= 0.0f) return x;
    auto mask = random::bernoulli(1.0f - cfg_.dropout, x.shape());
    float scale = 1.0f / (1.0f - cfg_.dropout);
    return x * where(mask, array(scale), array(0.0f));
}

array GPT::apply_rope(const array& x, const array& cos, const array& sin) {
    int half = head_dim_ / 2;
    auto parts = split(x, Shape{half}, -1);
    auto rot = concatenate({-parts[1], parts[0]}, -1);
    return x * cos + rot * sin;
}

array GPT::make_mask(const array& idx) {
    int T = idx.shape(-1);
    auto eos = equal(idx, array(cfg_.eos_id));
    auto eos_f = astype(eos, float32);
    auto seg = cumsum(eos_f, -1) - eos_f;
    auto same_seg = equal(expand_dims(seg, -1), expand_dims(seg, -2));
    auto causal = tril(ones({T, T}, bool_));
    auto allowed = logical_and(causal, same_seg);
    auto mask = where(allowed, array(0.0f), array(-1e9f));
    return expand_dims(mask, 1);
}

array GPT::forward(const array& idx, bool train) {
    int B = idx.shape(0);
    int T = idx.shape(1);

    auto x = take(token_emb_, idx, 0);
    if (train) x = dropout_(x);

    // Slice cached RoPE to current seq_len
    auto cos_v = slice(rope_cos_, {0, 0, 0, 0}, {1, 1, T, head_dim_});
    auto sin_v = slice(rope_sin_, {0, 0, 0, 0}, {1, 1, T, head_dim_});
    auto mask = make_mask(idx);

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (auto& b : blocks_) {
        auto h = rmsnorm(x, b.attn_norm_w);

        auto h_flat = reshape(h, {B * T, cfg_.d_model});
        auto qkv_flat = matmul(h_flat, transpose(b.qkv_w, {1, 0}));
        auto qkv = reshape(qkv_flat, {B, T, 3 * cfg_.d_model});
        auto qkv_parts = split(qkv, 3, -1);
        auto q = reshape(qkv_parts[0], {B, T, cfg_.n_head, head_dim_});
        auto k = reshape(qkv_parts[1], {B, T, cfg_.n_head, head_dim_});
        auto v = reshape(qkv_parts[2], {B, T, cfg_.n_head, head_dim_});

        q = transpose(q, {0, 2, 1, 3});
        k = transpose(k, {0, 2, 1, 3});
        v = transpose(v, {0, 2, 1, 3});

        q = apply_rope(q, cos_v, sin_v);
        k = apply_rope(k, cos_v, sin_v);

        if (cfg_.qk_norm) {
            q = rmsnorm(q, b.q_norm_w);
            k = rmsnorm(k, b.k_norm_w);
        }

        array attn_out({0.0f}, float32);
        if (train) {
            // Manual attention: needed for attention dropout
            auto scores = matmul(q, transpose(k, {0, 1, 3, 2})) * attn_scale + mask;
            auto weights = softmax(scores, -1);
            weights = dropout_(weights);
            attn_out = matmul(weights, v);
        } else {
            // SDPA: fused kernel, no dropout needed for eval/inference
            attn_out = fast::scaled_dot_product_attention(q, k, v, attn_scale, "", mask);
        }

        attn_out = transpose(attn_out, {0, 2, 1, 3});
        attn_out = reshape(attn_out, {B * T, cfg_.d_model});
        attn_out = matmul(attn_out, transpose(b.proj_w, {1, 0}));
        attn_out = reshape(attn_out, {B, T, cfg_.d_model});
        if (train) attn_out = dropout_(attn_out);

        auto m = rmsnorm(x, b.mlp_norm_w);
        auto m_flat = reshape(m, {B * T, cfg_.d_model});
        auto gate_flat = matmul(m_flat, transpose(b.w_gate, {1, 0}));
        auto up_flat = matmul(m_flat, transpose(b.w_up, {1, 0}));
        auto gate = reshape(gate_flat, {B, T, hidden_});
        auto up = reshape(up_flat, {B, T, hidden_});
        auto silu_gate = gate * sigmoid(gate);
        auto combined_flat = reshape(silu_gate * up, {B * T, hidden_});
        auto mlp_out_flat = matmul(combined_flat, transpose(b.w_out, {1, 0}));
        auto mlp_out = reshape(mlp_out_flat, {B, T, cfg_.d_model});
        if (train) mlp_out = dropout_(mlp_out);

        x = x + attn_out + mlp_out;
    }

    x = rmsnorm(x, ln_f_w_);
    auto x_flat = reshape(x, {B * T, cfg_.d_model});
    auto logits_flat = matmul(x_flat, transpose(token_emb_, {1, 0}));
    auto logits = reshape(logits_flat, {B, T, cfg_.vocab_size});
    return logits;
}

array GPT::loss(const array& idx, const array& targets, bool train) {
    auto logits = forward(idx, train);
    auto lse = logsumexp(logits, -1, true);
    auto log_probs = logits - lse;
    auto tgt = expand_dims(targets, -1);
    auto picked = squeeze(take_along_axis(log_probs, tgt, -1), -1);
    return -mean(picked);
}

array GPT::forward_cached(const array& idx,
                            std::vector<std::pair<array, array>>& kv_cache,
                            int prev_len) {
    int B = idx.shape(0);
    int T = idx.shape(1);

    auto x = take(token_emb_, idx, 0);

    // RoPE for positions [prev_len, prev_len+T)
    int hd = head_dim_;
    int half = hd / 2;
    std::vector<float> inv_freq_data(half);
    for (int i = 0; i < half; i++)
        inv_freq_data[i] = 1.0f / std::pow(cfg_.rope_theta, 2.0f * i / hd);
    array inv_freq(inv_freq_data.begin(), {half}, float32);
    auto positions = arange((float)prev_len, (float)(prev_len + T), float32);
    auto freqs = outer(positions, inv_freq);
    auto emb = concatenate({freqs, freqs}, -1);
    auto cos_v = reshape(cos(emb), {1, 1, T, hd});
    auto sin_v = reshape(sin(emb), {1, 1, T, hd});

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));

    for (int li = 0; li < (int)blocks_.size(); li++) {
        auto& b = blocks_[li];
        auto h = rmsnorm(x, b.attn_norm_w);

        auto h_flat = reshape(h, {B * T, cfg_.d_model});
        auto qkv_flat = matmul(h_flat, transpose(b.qkv_w, {1, 0}));
        auto qkv = reshape(qkv_flat, {B, T, 3 * cfg_.d_model});
        auto qkv_parts = split(qkv, 3, -1);
        auto q = reshape(qkv_parts[0], {B, T, cfg_.n_head, hd});
        auto k = reshape(qkv_parts[1], {B, T, cfg_.n_head, hd});
        auto v = reshape(qkv_parts[2], {B, T, cfg_.n_head, hd});

        q = transpose(q, {0, 2, 1, 3});
        k = transpose(k, {0, 2, 1, 3});
        v = transpose(v, {0, 2, 1, 3});

        q = apply_rope(q, cos_v, sin_v);
        k = apply_rope(k, cos_v, sin_v);

        if (cfg_.qk_norm) {
            q = rmsnorm(q, b.q_norm_w);
            k = rmsnorm(k, b.k_norm_w);
        }

        // Append k,v to cache
        if (prev_len == 0) {
            kv_cache[li].first = k;
            kv_cache[li].second = v;
        } else {
            kv_cache[li].first = concatenate({kv_cache[li].first, k}, 2);
            kv_cache[li].second = concatenate({kv_cache[li].second, v}, 2);
        }
        auto& k_cached = kv_cache[li].first;
        auto& v_cached = kv_cache[li].second;

        // Attention: q [B,H,T,d] @ k_cached [B,H,S,d] -> [B,H,T,S]
        auto scores = matmul(q, transpose(k_cached, {0, 1, 3, 2})) * attn_scale;
        // Causal mask: position i can attend to positions 0..prev_len+i
        int S = k_cached.shape(2);
        auto causal = tril(ones({T, S}, bool_));
        auto mask_vals = where(causal, array(0.0f), array(-1e9f));
        // [T, S] -> [1, 1, T, S] to broadcast over B and H
        auto mask = reshape(mask_vals, {1, 1, T, S});
        auto weights = softmax(scores + mask, -1);
        auto attn_out = matmul(weights, v_cached);

        attn_out = transpose(attn_out, {0, 2, 1, 3});
        attn_out = reshape(attn_out, {B * T, cfg_.d_model});
        attn_out = matmul(attn_out, transpose(b.proj_w, {1, 0}));
        attn_out = reshape(attn_out, {B, T, cfg_.d_model});

        auto m = rmsnorm(x, b.mlp_norm_w);
        auto m_flat = reshape(m, {B * T, cfg_.d_model});
        auto gate_flat = matmul(m_flat, transpose(b.w_gate, {1, 0}));
        auto up_flat = matmul(m_flat, transpose(b.w_up, {1, 0}));
        auto gate = reshape(gate_flat, {B, T, hidden_});
        auto up = reshape(up_flat, {B, T, hidden_});
        auto silu_gate = gate * sigmoid(gate);
        auto combined_flat = reshape(silu_gate * up, {B * T, hidden_});
        auto mlp_out_flat = matmul(combined_flat, transpose(b.w_out, {1, 0}));
        auto mlp_out = reshape(mlp_out_flat, {B, T, cfg_.d_model});

        x = x + attn_out + mlp_out;
    }

    x = rmsnorm(x, ln_f_w_);
    auto x_flat = reshape(x, {B * T, cfg_.d_model});
    auto logits_flat = matmul(x_flat, transpose(token_emb_, {1, 0}));
    auto logits = reshape(logits_flat, {B, T, cfg_.vocab_size});
    return logits;
}

std::vector<array*> GPT::parameters() {
    std::vector<array*> params;
    params.push_back(&token_emb_);
    for (auto& b : blocks_) {
        params.push_back(&b.attn_norm_w);
        params.push_back(&b.qkv_w);
        params.push_back(&b.q_norm_w);
        params.push_back(&b.k_norm_w);
        params.push_back(&b.proj_w);
        params.push_back(&b.mlp_norm_w);
        params.push_back(&b.w_gate);
        params.push_back(&b.w_up);
        params.push_back(&b.w_out);
    }
    params.push_back(&ln_f_w_);
    return params;
}

void GPT::set_parameters(const std::vector<array>& params) {
    size_t i = 0;
    token_emb_ = params[i++];
    for (auto& b : blocks_) {
        b.attn_norm_w = params[i++];
        b.qkv_w = params[i++];
        b.q_norm_w = params[i++];
        b.k_norm_w = params[i++];
        b.proj_w = params[i++];
        b.mlp_norm_w = params[i++];
        b.w_gate = params[i++];
        b.w_up = params[i++];
        b.w_out = params[i++];
    }
    ln_f_w_ = params[i++];
}
