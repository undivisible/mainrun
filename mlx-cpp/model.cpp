#include "model.h"

#include <cmath>
#include <random>

using namespace mlx::core;

static array normal_array(std::mt19937& gen, float mean, float std, Shape shape) {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    std::vector<float> data(n);
    std::normal_distribution<float> dist(mean, std);
    for (auto& v : data) v = dist(gen);
    return array(data.begin(), std::move(shape), float32);
}

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

array GPT::rmsnorm(const array& x, const array& w) const {
    return fast::rms_norm(x, w, cfg_.rms_eps);
}

array GPT::dropout_(const array& x) const {
    if (cfg_.dropout <= 0.0f) return x;
    auto mask = random::bernoulli(1.0f - cfg_.dropout, x.shape());
    float scale = 1.0f / (1.0f - cfg_.dropout);
    return x * where(mask, array(scale), array(0.0f));
}

array GPT::make_mask(const array& idx) const {
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

// Compiled SwiGLU: gate * sigmoid(gate) * up — fused Metal kernel
static auto compiled_swiglu = compile([](const std::vector<array>& inputs) -> std::vector<array> {
    const auto& gate = inputs[0];
    const auto& up = inputs[1];
    return {multiply(multiply(gate, sigmoid(gate)), up)};
});

array GPT::forward(const array& idx, bool train) {
    int B = idx.shape(0);
    int T = idx.shape(1);

    auto x = take(token_emb_, idx, 0);
    if (train) x = dropout_(x);

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

        q = fast::rope(q, head_dim_, false, cfg_.rope_theta, 1.0f, 0);
        k = fast::rope(k, head_dim_, false, cfg_.rope_theta, 1.0f, 0);

        if (cfg_.qk_norm) {
            q = rmsnorm(q, b.q_norm_w);
            k = rmsnorm(k, b.k_norm_w);
        }

        auto attn_out = fast::scaled_dot_product_attention(q, k, v, attn_scale, "", mask);

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
        auto silu_result = train
            ? multiply(multiply(gate, sigmoid(gate)), up)
            : compiled_swiglu({gate, up})[0];
        auto combined_flat = reshape(silu_result, {B * T, hidden_});
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

array GPT::forward_functional(const std::vector<array>& params, const array& idx, bool train) const {
    int B = idx.shape(0);
    int T = idx.shape(1);

    size_t pi = 0;
    const array& token_emb = params[pi++];

    auto x = take(token_emb, idx, 0);
    if (train) x = dropout_(x);

    auto cos_v = slice(rope_cos_, {0, 0, 0, 0}, {1, 1, T, head_dim_});
    auto sin_v = slice(rope_sin_, {0, 0, 0, 0}, {1, 1, T, head_dim_});
    auto mask = make_mask(idx);

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (size_t li = 0; li < blocks_.size(); li++) {
        const array& attn_norm_w = params[pi++];
        const array& qkv_w = params[pi++];
        const array& q_norm_w = params[pi++];
        const array& k_norm_w = params[pi++];
        const array& proj_w = params[pi++];
        const array& mlp_norm_w = params[pi++];
        const array& w_gate = params[pi++];
        const array& w_up = params[pi++];
        const array& w_out = params[pi++];

        auto h = rmsnorm(x, attn_norm_w);

        auto h_flat = reshape(h, {B * T, cfg_.d_model});
        auto qkv_flat = matmul(h_flat, transpose(qkv_w, {1, 0}));
        auto qkv = reshape(qkv_flat, {B, T, 3 * cfg_.d_model});
        auto qkv_parts = split(qkv, 3, -1);
        auto q = reshape(qkv_parts[0], {B, T, cfg_.n_head, head_dim_});
        auto k = reshape(qkv_parts[1], {B, T, cfg_.n_head, head_dim_});
        auto v = reshape(qkv_parts[2], {B, T, cfg_.n_head, head_dim_});

        q = transpose(q, {0, 2, 1, 3});
        k = transpose(k, {0, 2, 1, 3});
        v = transpose(v, {0, 2, 1, 3});

        q = fast::rope(q, head_dim_, false, cfg_.rope_theta, 1.0f, 0);
        k = fast::rope(k, head_dim_, false, cfg_.rope_theta, 1.0f, 0);

        if (cfg_.qk_norm) {
            q = rmsnorm(q, q_norm_w);
            k = rmsnorm(k, k_norm_w);
        }

        auto attn_out = fast::scaled_dot_product_attention(q, k, v, attn_scale, "", mask);

        attn_out = transpose(attn_out, {0, 2, 1, 3});
        attn_out = reshape(attn_out, {B * T, cfg_.d_model});
        attn_out = matmul(attn_out, transpose(proj_w, {1, 0}));
        attn_out = reshape(attn_out, {B, T, cfg_.d_model});
        if (train) attn_out = dropout_(attn_out);

        auto m = rmsnorm(x, mlp_norm_w);
        auto m_flat = reshape(m, {B * T, cfg_.d_model});
        auto gate_flat = matmul(m_flat, transpose(w_gate, {1, 0}));
        auto up_flat = matmul(m_flat, transpose(w_up, {1, 0}));
        auto gate = reshape(gate_flat, {B, T, hidden_});
        auto up = reshape(up_flat, {B, T, hidden_});
        auto silu_result = train
            ? multiply(multiply(gate, sigmoid(gate)), up)
            : compiled_swiglu({gate, up})[0];
        auto combined_flat = reshape(silu_result, {B * T, hidden_});
        auto mlp_out_flat = matmul(combined_flat, transpose(w_out, {1, 0}));
        auto mlp_out = reshape(mlp_out_flat, {B, T, cfg_.d_model});
        if (train) mlp_out = dropout_(mlp_out);

        x = x + attn_out + mlp_out;
    }

    const array& ln_f_w = params[pi++];
    x = rmsnorm(x, ln_f_w);
    auto x_flat = reshape(x, {B * T, cfg_.d_model});
    auto logits_flat = matmul(x_flat, transpose(token_emb, {1, 0}));
    auto logits = reshape(logits_flat, {B, T, cfg_.vocab_size});
    return logits;
}

array GPT::loss_functional(const std::vector<array>& params, const array& idx, const array& targets, bool train) const {
    auto logits = forward_functional(params, idx, train);
    auto lse = logsumexp(logits, -1, true);
    auto log_probs = logits - lse;
    auto tgt = expand_dims(targets, -1);
    auto picked = squeeze(take_along_axis(log_probs, tgt, -1), -1);
    return -mean(picked);
}

// --- Quantization helpers ---

GPT::QuantW GPT::quantize_weight(const array& w) {
    auto result = quantize(w, 64, 4, "affine");
    eval(result);
    return {result[0], result[1], result[2]};
}

array GPT::qmatmul(const array& x, const QuantW& qw) {
    return quantized_matmul(x, qw.w, qw.scales, qw.biases, true);
}

void GPT::quantize_for_inference() {
    q_emb_ = quantize_weight(token_emb_);
    q_qkv_.clear(); q_proj_.clear(); q_gate_.clear(); q_up_.clear(); q_out_.clear();
    for (auto& b : blocks_) {
        q_qkv_.push_back(quantize_weight(b.qkv_w));
        q_proj_.push_back(quantize_weight(b.proj_w));
        q_gate_.push_back(quantize_weight(b.w_gate));
        q_up_.push_back(quantize_weight(b.w_up));
        q_out_.push_back(quantize_weight(b.w_out));
    }
    quantized_ = true;
    fprintf(stderr, "Quantized %d layers for inference (4-bit, group=64)\n", (int)q_qkv_.size());
}

// --- Inference dtype promotion ------------------------------------------------

void GPT::promote_for_inference(Dtype dtype) {
    if (dtype != float32 && dtype != bfloat16 && dtype != float16) {
        fprintf(stderr, "promote_for_inference: unsupported dtype, keeping float32\n");
        return;
    }
    token_emb_ = astype(token_emb_, dtype);
    for (auto& b : blocks_) {
        b.attn_norm_w = astype(b.attn_norm_w, dtype);
        b.qkv_w = astype(b.qkv_w, dtype);
        b.q_norm_w = astype(b.q_norm_w, dtype);
        b.k_norm_w = astype(b.k_norm_w, dtype);
        b.proj_w = astype(b.proj_w, dtype);
        b.mlp_norm_w = astype(b.mlp_norm_w, dtype);
        b.w_gate = astype(b.w_gate, dtype);
        b.w_up = astype(b.w_up, dtype);
        b.w_out = astype(b.w_out, dtype);
    }
    ln_f_w_ = astype(ln_f_w_, dtype);
    eval(token_emb_, ln_f_w_);
    for (auto& b : blocks_) {
        eval(b.attn_norm_w, b.qkv_w, b.q_norm_w, b.k_norm_w, b.proj_w,
             b.mlp_norm_w, b.w_gate, b.w_up, b.w_out);
    }
    infer_dtype_ = dtype;
    fprintf(stderr, "Promoted parameters to %s for inference\n",
            dtype == bfloat16 ? "bfloat16" : (dtype == float16 ? "float16" : "float32"));
}

// --- Compiled decode step (T=1, stable KV cache shapes) -----------------------

std::vector<array> GPT::decode_step_impl(const std::vector<array>& inputs) {
    const auto& idx = inputs[0];
    auto k_cache = inputs[1];
    auto v_cache = inputs[2];
    auto prev_len = inputs[3];

    int B = idx.shape(0);
    int T = idx.shape(1);
    int V = cfg_.vocab_size;

    auto x = take(token_emb_, idx, 0);

    // Position mask for the full block_size cache: positions 0..prev_len valid.
    auto positions = arange(0, cfg_.block_size, int32);
    auto valid = less_equal(positions, prev_len);
    auto mask1d = where(valid, array(0.0f, infer_dtype_), array(-1e9f, infer_dtype_));
    auto mask = expand_dims(mask1d, {0, 1, 2});

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (int li = 0; li < (int)blocks_.size(); li++) {
        auto& b = blocks_[li];
        auto h = rmsnorm(x, b.attn_norm_w);

        auto h_flat = reshape(h, {B * T, cfg_.d_model});

        auto qkv_flat = quantized_
            ? qmatmul(h_flat, q_qkv_[li])
            : matmul(h_flat, transpose(b.qkv_w, {1, 0}));
        auto qkv = reshape(qkv_flat, {B, T, 3 * cfg_.d_model});
        auto qkv_parts = split(qkv, 3, -1);
        auto q = reshape(qkv_parts[0], {B, T, cfg_.n_head, head_dim_});
        auto k = reshape(qkv_parts[1], {B, T, cfg_.n_head, head_dim_});
        auto v = reshape(qkv_parts[2], {B, T, cfg_.n_head, head_dim_});

        q = transpose(q, {0, 2, 1, 3});
        k = transpose(k, {0, 2, 1, 3});
        v = transpose(v, {0, 2, 1, 3});

        q = fast::rope(q, head_dim_, false, cfg_.rope_theta, 1.0f, prev_len);
        k = fast::rope(k, head_dim_, false, cfg_.rope_theta, 1.0f, prev_len);

        if (cfg_.qk_norm) {
            q = rmsnorm(q, b.q_norm_w);
            k = rmsnorm(k, b.k_norm_w);
        }

        k_cache = scatter(k_cache, prev_len, k, 2);
        v_cache = scatter(v_cache, prev_len, v, 2);

        auto attn_out = fast::scaled_dot_product_attention(
            q, k_cache, v_cache, attn_scale, "", mask);

        attn_out = transpose(attn_out, {0, 2, 1, 3});
        attn_out = reshape(attn_out, {B * T, cfg_.d_model});

        auto proj_out = quantized_
            ? qmatmul(attn_out, q_proj_[li])
            : matmul(attn_out, transpose(b.proj_w, {1, 0}));
        attn_out = reshape(proj_out, {B, T, cfg_.d_model});

        auto m = rmsnorm(x, b.mlp_norm_w);
        auto m_flat = reshape(m, {B * T, cfg_.d_model});

        auto gate_flat = quantized_
            ? qmatmul(m_flat, q_gate_[li])
            : matmul(m_flat, transpose(b.w_gate, {1, 0}));
        auto up_flat = quantized_
            ? qmatmul(m_flat, q_up_[li])
            : matmul(m_flat, transpose(b.w_up, {1, 0}));
        auto gate = reshape(gate_flat, {B, T, hidden_});
        auto up = reshape(up_flat, {B, T, hidden_});
        auto silu_result = compiled_swiglu({gate, up})[0];
        auto combined_flat = reshape(silu_result, {B * T, hidden_});

        auto mlp_out_flat = quantized_
            ? qmatmul(combined_flat, q_out_[li])
            : matmul(combined_flat, transpose(b.w_out, {1, 0}));
        auto mlp_out = reshape(mlp_out_flat, {B, T, cfg_.d_model});

        x = x + attn_out + mlp_out;
    }

    x = rmsnorm(x, ln_f_w_);
    auto x_flat = reshape(x, {B * T, cfg_.d_model});

    auto logits_flat = quantized_
        ? qmatmul(x_flat, q_emb_)
        : matmul(x_flat, transpose(token_emb_, {1, 0}));
    auto logits = reshape(logits_flat, {B, T, V});

    return {logits, k_cache, v_cache};
}

// --- KV-cached forward with quantized weights + fast kernels ------------------

array GPT::forward_cached(const array& idx, array& k_cache, array& v_cache, int prev_len) {
    int B = idx.shape(0);
    int T = idx.shape(1);

    // Non-quantized T=1 decode: use the compiled step with stable cache shapes.
    if (T == 1 && !quantized_) {
        auto& compiled = compiled_decode_fp32_;
        if (!compiled) {
            std::function<std::vector<array>(const std::vector<array>&)> step =
                [this](const std::vector<array>& inputs) -> std::vector<array> {
                return decode_step_impl(inputs);
            };
            compiled = compile(step, false);
        }
        auto out = compiled({idx, k_cache, v_cache, array(prev_len)});
        k_cache = out[1];
        v_cache = out[2];
        return out[0];
    }

    // Multi-token prefill / decode: write into the stable cache and attend
    // causally over the slice of valid positions.
    auto x = take(token_emb_, idx, 0);
    auto idx_range = arange(0, T, int32);

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (int li = 0; li < (int)blocks_.size(); li++) {
        auto& b = blocks_[li];
        auto h = rmsnorm(x, b.attn_norm_w);

        auto h_flat = reshape(h, {B * T, cfg_.d_model});

        auto qkv_flat = quantized_
            ? qmatmul(h_flat, q_qkv_[li])
            : matmul(h_flat, transpose(b.qkv_w, {1, 0}));
        auto qkv = reshape(qkv_flat, {B, T, 3 * cfg_.d_model});
        auto qkv_parts = split(qkv, 3, -1);
        auto q = reshape(qkv_parts[0], {B, T, cfg_.n_head, head_dim_});
        auto k = reshape(qkv_parts[1], {B, T, cfg_.n_head, head_dim_});
        auto v = reshape(qkv_parts[2], {B, T, cfg_.n_head, head_dim_});

        q = transpose(q, {0, 2, 1, 3});
        k = transpose(k, {0, 2, 1, 3});
        v = transpose(v, {0, 2, 1, 3});

        q = fast::rope(q, head_dim_, false, cfg_.rope_theta, 1.0f, prev_len);
        k = fast::rope(k, head_dim_, false, cfg_.rope_theta, 1.0f, prev_len);

        if (cfg_.qk_norm) {
            q = rmsnorm(q, b.q_norm_w);
            k = rmsnorm(k, b.k_norm_w);
        }

        // Reshape K/V so the leading dimension is the index length and the
        // remaining dimensions match the cache slice shape for scatter.
        auto k_for_scatter = reshape(transpose(k, {2, 0, 1, 3}), {T, 1, cfg_.n_head, 1, head_dim_});
        auto v_for_scatter = reshape(transpose(v, {2, 0, 1, 3}), {T, 1, cfg_.n_head, 1, head_dim_});
        k_cache = scatter(k_cache, idx_range, k_for_scatter, 2);
        v_cache = scatter(v_cache, idx_range, v_for_scatter, 2);

        auto k_sliced = slice(k_cache, {0, 0, 0, 0}, {B, cfg_.n_head, prev_len + T, head_dim_});
        auto v_sliced = slice(v_cache, {0, 0, 0, 0}, {B, cfg_.n_head, prev_len + T, head_dim_});

        auto attn_out = fast::scaled_dot_product_attention(q, k_sliced, v_sliced, attn_scale, "causal");

        attn_out = transpose(attn_out, {0, 2, 1, 3});
        attn_out = reshape(attn_out, {B * T, cfg_.d_model});

        auto proj_out = quantized_
            ? qmatmul(attn_out, q_proj_[li])
            : matmul(attn_out, transpose(b.proj_w, {1, 0}));
        attn_out = reshape(proj_out, {B, T, cfg_.d_model});

        auto m = rmsnorm(x, b.mlp_norm_w);
        auto m_flat = reshape(m, {B * T, cfg_.d_model});

        auto gate_flat = quantized_
            ? qmatmul(m_flat, q_gate_[li])
            : matmul(m_flat, transpose(b.w_gate, {1, 0}));
        auto up_flat = quantized_
            ? qmatmul(m_flat, q_up_[li])
            : matmul(m_flat, transpose(b.w_up, {1, 0}));
        auto gate = reshape(gate_flat, {B, T, hidden_});
        auto up = reshape(up_flat, {B, T, hidden_});
        auto silu_result = compiled_swiglu({gate, up})[0];
        auto combined_flat = reshape(silu_result, {B * T, hidden_});

        auto mlp_out_flat = quantized_
            ? qmatmul(combined_flat, q_out_[li])
            : matmul(combined_flat, transpose(b.w_out, {1, 0}));
        auto mlp_out = reshape(mlp_out_flat, {B, T, cfg_.d_model});

        x = x + attn_out + mlp_out;
    }

    x = rmsnorm(x, ln_f_w_);
    auto x_flat = reshape(x, {B * T, cfg_.d_model});

    auto logits_flat = quantized_
        ? qmatmul(x_flat, q_emb_)
        : matmul(x_flat, transpose(token_emb_, {1, 0}));
    auto logits = reshape(logits_flat, {B, T, cfg_.vocab_size});
    return logits;
}

// --- Growing-cache decode (T=1) for quantized fast path ----------------------

array GPT::forward_decode_growing(
    const array& idx,
    std::vector<std::pair<array, array>>& kv_cache,
    int prev_len) {
    int B = idx.shape(0);
    int T = idx.shape(1);

    auto x = take(token_emb_, idx, 0);
    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (int li = 0; li < (int)blocks_.size(); li++) {
        auto& b = blocks_[li];
        auto h = rmsnorm(x, b.attn_norm_w);

        auto h_flat = reshape(h, {B * T, cfg_.d_model});
        auto qkv_flat = quantized_
            ? qmatmul(h_flat, q_qkv_[li])
            : matmul(h_flat, transpose(b.qkv_w, {1, 0}));
        auto qkv = reshape(qkv_flat, {B, T, 3 * cfg_.d_model});
        auto qkv_parts = split(qkv, 3, -1);
        auto q = reshape(qkv_parts[0], {B, T, cfg_.n_head, head_dim_});
        auto k = reshape(qkv_parts[1], {B, T, cfg_.n_head, head_dim_});
        auto v = reshape(qkv_parts[2], {B, T, cfg_.n_head, head_dim_});

        q = transpose(q, {0, 2, 1, 3});
        k = transpose(k, {0, 2, 1, 3});
        v = transpose(v, {0, 2, 1, 3});

        q = fast::rope(q, head_dim_, false, cfg_.rope_theta, 1.0f, prev_len);
        k = fast::rope(k, head_dim_, false, cfg_.rope_theta, 1.0f, prev_len);

        if (cfg_.qk_norm) {
            q = rmsnorm(q, b.q_norm_w);
            k = rmsnorm(k, b.k_norm_w);
        }

        if (prev_len == 0) {
            kv_cache[li].first = k;
            kv_cache[li].second = v;
        } else {
            kv_cache[li].first = concatenate({kv_cache[li].first, k}, 2);
            kv_cache[li].second = concatenate({kv_cache[li].second, v}, 2);
        }
        auto& k_cached = kv_cache[li].first;
        auto& v_cached = kv_cache[li].second;

        auto attn_out = fast::scaled_dot_product_attention(
            q, k_cached, v_cached, attn_scale, "causal");

        attn_out = transpose(attn_out, {0, 2, 1, 3});
        attn_out = reshape(attn_out, {B * T, cfg_.d_model});

        auto proj_out = quantized_
            ? qmatmul(attn_out, q_proj_[li])
            : matmul(attn_out, transpose(b.proj_w, {1, 0}));
        attn_out = reshape(proj_out, {B, T, cfg_.d_model});

        auto m = rmsnorm(x, b.mlp_norm_w);
        auto m_flat = reshape(m, {B * T, cfg_.d_model});

        auto gate_flat = quantized_
            ? qmatmul(m_flat, q_gate_[li])
            : matmul(m_flat, transpose(b.w_gate, {1, 0}));
        auto up_flat = quantized_
            ? qmatmul(m_flat, q_up_[li])
            : matmul(m_flat, transpose(b.w_up, {1, 0}));
        auto gate = reshape(gate_flat, {B, T, hidden_});
        auto up = reshape(up_flat, {B, T, hidden_});
        auto silu_result = compiled_swiglu({gate, up})[0];
        auto combined_flat = reshape(silu_result, {B * T, hidden_});

        auto mlp_out_flat = quantized_
            ? qmatmul(combined_flat, q_out_[li])
            : matmul(combined_flat, transpose(b.w_out, {1, 0}));
        auto mlp_out = reshape(mlp_out_flat, {B, T, cfg_.d_model});

        x = x + attn_out + mlp_out;
    }

    x = rmsnorm(x, ln_f_w_);
    auto x_flat = reshape(x, {B * T, cfg_.d_model});

    auto logits_flat = quantized_
        ? qmatmul(x_flat, q_emb_)
        : matmul(x_flat, transpose(token_emb_, {1, 0}));
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
