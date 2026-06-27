#include "inference.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>

#include <mlx/mlx.h>

namespace mx = mlx::core;

// ---------------------------------------------------------------------------
// Checkpoint save / load
// ---------------------------------------------------------------------------

static std::vector<std::string> param_names(const GPTConfig& cfg) {
    std::vector<std::string> names;
    names.reserve(1 + cfg.n_layer * 9 + 1);
    names.push_back("token_emb.weight");
    for (int i = 0; i < cfg.n_layer; ++i) {
        names.push_back("blocks." + std::to_string(i) + ".attn_norm.weight");
        names.push_back("blocks." + std::to_string(i) + ".qkv.weight");
        names.push_back("blocks." + std::to_string(i) + ".q_norm.weight");
        names.push_back("blocks." + std::to_string(i) + ".k_norm.weight");
        names.push_back("blocks." + std::to_string(i) + ".proj.weight");
        names.push_back("blocks." + std::to_string(i) + ".mlp_norm.weight");
        names.push_back("blocks." + std::to_string(i) + ".w_gate");
        names.push_back("blocks." + std::to_string(i) + ".w_up");
        names.push_back("blocks." + std::to_string(i) + ".w_out");
    }
    names.push_back("ln_f.weight");
    return names;
}

void save_checkpoint(const std::string& path, GPT& model) {
    auto ptrs = model.parameters();
    auto names = param_names(model.config());
    if (ptrs.size() != names.size()) {
        throw std::runtime_error("save_checkpoint: parameter count mismatch");
    }
    std::unordered_map<std::string, mx::array> map;
    for (size_t i = 0; i < ptrs.size(); ++i) {
        map.emplace(names[i], *ptrs[i]);
    }
    mx::save_safetensors(path, map);
}

void load_checkpoint(const std::string& path, GPT& model) {
    auto [arrays, meta] = mx::load_safetensors(path);
    auto names = param_names(model.config());
    std::vector<mx::array> params;
    params.reserve(names.size());
    for (const auto& name : names) {
        auto it = arrays.find(name);
        if (it == arrays.end()) {
            throw std::runtime_error("load_checkpoint: missing key " + name);
        }
        params.push_back(it->second);
    }
    model.set_parameters(params);
}

// ---------------------------------------------------------------------------
// Tokenizer via Python subprocess
// ---------------------------------------------------------------------------

static const char* tok_script_path = "/tmp/mlx_cpp_tok.py";

static std::string find_python() {
    // Try venv python first, then system python3
    const char* candidates[] = {
        "../mainrun/.venv/bin/python3",
        "mainrun/.venv/bin/python3",
        "../../mainrun/.venv/bin/python3",
        "python3",
    };
    for (auto& p : candidates) {
        std::string cmd = std::string(p) + " -c 'import tokenizers' 2>/dev/null";
        if (std::system(cmd.c_str()) == 0) return p;
    }
    return "python3";
}

static void ensure_tok_script() {
    static bool written = false;
    if (written) return;
    const char* script =
        "import sys, json\n"
        "from tokenizers import Tokenizer\n"
        "tok = Tokenizer.from_file(sys.argv[1])\n"
        "if sys.argv[2] == 'encode':\n"
        "    ids = tok.encode(sys.argv[3]).ids\n"
        "    print(json.dumps(ids))\n"
        "elif sys.argv[2] == 'decode':\n"
        "    ids = json.loads(sys.argv[3])\n"
        "    print(tok.decode(ids, skip_special_tokens=True))\n";
    FILE* f = std::fopen(tok_script_path, "w");
    if (!f) throw std::runtime_error("ensure_tok_script: cannot write helper");
    std::fputs(script, f);
    std::fclose(f);
    written = true;
}

// ponytail: shell-escape a string for safe single-quoted argv
static std::string shell_escape(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += '\'';
    return out;
}

static std::string run_capture(const std::string& cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("run_capture: popen failed for: " + cmd);
    std::string out;
    char buf[4096];
    while (size_t n = std::fread(buf, 1, sizeof(buf), pipe)) {
        out.append(buf, n);
    }
    int rc = pclose(pipe);
    if (rc != 0) throw std::runtime_error("run_capture: command failed (rc=" +
                                          std::to_string(rc) + "): " + out);
    return out;
}

std::vector<uint32_t> encode_text(const std::string& text, const std::string& tokenizer_path) {
    ensure_tok_script();
    static std::string py = find_python();
    std::string cmd = py + " " + std::string(tok_script_path) + " " +
                      shell_escape(tokenizer_path) + " encode " + shell_escape(text);
    std::string out = run_capture(cmd);
    // parse JSON array of ints
    std::vector<uint32_t> ids;
    std::string s = out;
    s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    if (s.size() < 2 || s.front() != '[' || s.back() != ']') {
        throw std::runtime_error("encode_text: unexpected output: " + out);
    }
    std::stringstream ss(s.substr(1, s.size() - 2));
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        ids.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    return ids;
}

std::string decode_ids(const std::vector<uint32_t>& ids, const std::string& tokenizer_path) {
    ensure_tok_script();
    static std::string py = find_python();
    std::string json_ids = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) json_ids += ",";
        json_ids += std::to_string(ids[i]);
    }
    json_ids += "]";
    std::string cmd = py + " " + std::string(tok_script_path) + " " +
                      shell_escape(tokenizer_path) + " decode " + shell_escape(json_ids);
    std::string out = run_capture(cmd);
    // strip trailing newline
    if (!out.empty() && out.back() == '\n') out.pop_back();
    return out;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

uint32_t InferenceEngine::sample_token(mx::array& logits, const SamplingConfig& sampling) {
    // logits: [V] float32
    int V = static_cast<int>(logits.shape()[0]);
    logits.eval();
    const float* ptr = logits.data<float>();
    std::vector<float> vals(ptr, ptr + V);

    // temperature
    float temp = sampling.temperature > 0.0f ? sampling.temperature : 1e-5f;
    for (auto& v : vals) v /= temp;

    // top-k: gather indices of top_k highest logits
    int k = std::min<int>(sampling.top_k, V);
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return vals[a] > vals[b]; });
    std::vector<int> top_idx(idx.begin(), idx.begin() + k);

    // softmax over top-k
    float maxv = vals[top_idx[0]];
    std::vector<float> probs(k);
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        float e = std::exp(vals[top_idx[i]] - maxv);
        probs[i] = e;
        sum += e;
    }
    for (auto& p : probs) p /= sum;

    // top-p (nucleus): keep smallest set whose cumulative prob >= top_p
    std::vector<int> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return probs[a] > probs[b]; });
    float cum = 0.0f;
    int cutoff = k;
    for (int i = 0; i < k; ++i) {
        cum += probs[order[i]];
        if (cum >= sampling.top_p) { cutoff = i + 1; break; }
    }
    // renormalize over the kept set
    float kept_sum = 0.0f;
    for (int i = 0; i < cutoff; ++i) kept_sum += probs[order[i]];
    // sample
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> uni(0.0f, kept_sum);
    float r = uni(rng);
    cum = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        cum += probs[order[i]];
        if (r <= cum) return static_cast<uint32_t>(top_idx[order[i]]);
    }
    return static_cast<uint32_t>(top_idx[order[cutoff - 1]]);
}

// ---------------------------------------------------------------------------
// InferenceEngine
// ---------------------------------------------------------------------------

InferenceEngine::InferenceEngine(const std::string& checkpoint_path,
                                 const std::string& tokenizer_path,
                                 const GPTConfig& config)
    : model_(config), tokenizer_path_(tokenizer_path) {
    block_size_ = config.block_size;
    load_checkpoint(checkpoint_path, model_);
}

std::string InferenceEngine::generate(const std::string& prompt, int max_tokens,
                                      const SamplingConfig& sampling) {
    auto tokens = encode_text(prompt, tokenizer_path_);
    if (tokens.empty()) tokens.push_back(static_cast<uint32_t>(model_.config().eos_id));

    int n_layer = model_.config().n_layer;
    std::vector<std::pair<mx::array, mx::array>> kv_cache;
    kv_cache.reserve(n_layer);
    for (int i = 0; i < n_layer; i++) kv_cache.emplace_back(mx::array({0.0f}, mx::float32), mx::array({0.0f}, mx::float32));
    int V = model_.config().vocab_size;

    // Prefill: process all prompt tokens at once
    int T = static_cast<int>(tokens.size());
    if (T > block_size_) {
        tokens.erase(tokens.begin(), tokens.end() - block_size_);
        T = block_size_;
    }
    mx::array idx(mx::array(tokens.data(), {1, T}, mx::uint32));
    mx::array logits = model_.forward_cached(idx, kv_cache, 0);
    mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, V}), {V});
    uint32_t next = sample_token(last, sampling);
    tokens.push_back(next);
    int cached_len = T;

    // Decode: one token at a time with KV cache
    for (int step = 1; step < max_tokens; ++step) {
        if (static_cast<int>(next) == model_.config().eos_id) break;
        if (cached_len >= block_size_) {
            kv_cache.clear();
            for (int i = 0; i < n_layer; i++) kv_cache.emplace_back(mx::array({0.0f}, mx::float32), mx::array({0.0f}, mx::float32));
            int start = static_cast<int>(tokens.size()) - block_size_;
            std::vector<uint32_t> window(tokens.begin() + start, tokens.end());
            T = block_size_;
            mx::array idx2(mx::array(window.data(), {1, T}, mx::uint32));
            logits = model_.forward_cached(idx2, kv_cache, 0);
            cached_len = T;
        } else {
            mx::array one(mx::array(&next, {1, 1}, mx::uint32));
            logits = model_.forward_cached(one, kv_cache, cached_len);
            cached_len += 1;
        }
        last = mx::reshape(mx::slice(logits, {0, 0, 0}, {1, 1, V}), {V});
        next = sample_token(last, sampling);
        tokens.push_back(next);
    }

    return decode_ids(tokens, tokenizer_path_);
}

BenchmarkResult InferenceEngine::benchmark(const std::string& prompt, int max_tokens,
                                           const SamplingConfig& sampling) {
    auto tokens = encode_text(prompt, tokenizer_path_);
    int prompt_tokens = static_cast<int>(tokens.size());
    if (tokens.empty()) tokens.push_back(static_cast<uint32_t>(model_.config().eos_id));

    int n_layer = model_.config().n_layer;
    std::vector<std::pair<mx::array, mx::array>> kv_cache;
    kv_cache.reserve(n_layer);
    for (int i = 0; i < n_layer; i++) kv_cache.emplace_back(mx::array({0.0f}, mx::float32), mx::array({0.0f}, mx::float32));
    int V = model_.config().vocab_size;

    using clock = std::chrono::steady_clock;

    // Prefill
    auto t0 = clock::now();
    int T = static_cast<int>(tokens.size());
    if (T > block_size_) {
        tokens.erase(tokens.begin(), tokens.end() - block_size_);
        T = block_size_;
    }
    mx::array idx(mx::array(tokens.data(), {1, T}, mx::uint32));
    mx::array logits = model_.forward_cached(idx, kv_cache, 0);
    mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, V}), {V});
    uint32_t next = sample_token(last, sampling);
    tokens.push_back(next);
    int cached_len = T;
    mx::eval(logits);
    auto t1 = clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int generated = 1;
    bool stopped = (static_cast<int>(next) == model_.config().eos_id);

    while (!stopped && generated < max_tokens) {
        if (cached_len >= block_size_) {
            kv_cache.clear();
            for (int i = 0; i < n_layer; i++) kv_cache.emplace_back(mx::array({0.0f}, mx::float32), mx::array({0.0f}, mx::float32));
            int start = static_cast<int>(tokens.size()) - block_size_;
            std::vector<uint32_t> window(tokens.begin() + start, tokens.end());
            T = block_size_;
            mx::array idx2(mx::array(window.data(), {1, T}, mx::uint32));
            logits = model_.forward_cached(idx2, kv_cache, 0);
            cached_len = T;
        } else {
            mx::array one(mx::array(&next, {1, 1}, mx::uint32));
            logits = model_.forward_cached(one, kv_cache, cached_len);
            cached_len += 1;
        }
        last = mx::reshape(mx::slice(logits, {0, 0, 0}, {1, 1, V}), {V});
        next = sample_token(last, sampling);
        tokens.push_back(next);
        ++generated;
        if (static_cast<int>(next) == model_.config().eos_id) stopped = true;
    }

    auto t2 = clock::now();
    double total_s = std::chrono::duration<double>(t2 - t0).count();
    double tps = generated > 0 ? generated / total_s : 0.0;

    return BenchmarkResult{
        prompt_tokens,
        generated,
        prefill_ms,
        total_s,
        tps,
    };
}
