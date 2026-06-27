#include "inference/engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/stat.h>

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

static void validate_path(const std::string& path) {
    if (path.empty()) throw std::runtime_error("empty path");
    if (path.find('\0') != std::string::npos) throw std::runtime_error("path contains null byte");
    if (path.size() > 4096) throw std::runtime_error("path too long");
}

void save_checkpoint(const std::string& path, GPT& model) {
    validate_path(path);
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
    validate_path(path);
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

static std::string tok_script_path;

static std::string find_python() {
    const char* candidates[] = {
        "../mainrun/.venv/bin/python3",
        "mainrun/.venv/bin/python3",
        "../../mainrun/.venv/bin/python3",
        "python3",
    };
    for (auto& p : candidates) {
        pid_t pid = fork();
        if (pid < 0) continue;
        if (pid == 0) {
            const char* argv[] = {p, "-c", "import tokenizers", nullptr};
            execvp(p, const_cast<char* const*>(argv));
            _exit(127);
        }
        int status = 0;
        if (waitpid(pid, &status, 0) > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0)
            return p;
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
    char template_path[] = "/tmp/mlx_cpp_tok_XXXXXX";
    int fd = mkstemp(template_path);
    if (fd < 0) throw std::runtime_error("ensure_tok_script: mkstemp failed");
    size_t len = std::strlen(script);
    ssize_t n = write(fd, script, len);
    close(fd);
    if (n != (ssize_t)len) {
        unlink(template_path);
        throw std::runtime_error("ensure_tok_script: write failed");
    }
    tok_script_path = template_path;
    written = true;
}

static std::string run_capture(const std::vector<std::string>& argv) {
    if (argv.empty()) throw std::runtime_error("run_capture: empty argv");
    int pipefd[2];
    if (pipe(pipefd) < 0) throw std::runtime_error("run_capture: pipe failed");
    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        throw std::runtime_error("run_capture: fork failed");
    }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        std::vector<char*> args;
        for (auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
        args.push_back(nullptr);
        execvp(argv[0].c_str(), args.data());
        _exit(127);
    }
    close(pipefd[1]);
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) out.append(buf, n);
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("run_capture: command failed");
    return out;
}

std::vector<uint32_t> encode_text(const std::string& text, const std::string& tokenizer_path) {
    ensure_tok_script();
    static std::string py = find_python();
    std::string out = run_capture({py, tok_script_path, tokenizer_path, "encode", text});
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
    std::string out = run_capture({py, tok_script_path, tokenizer_path, "decode", json_ids});
    if (!out.empty() && out.back() == '\n') out.pop_back();
    return out;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

uint32_t InferenceEngine::sample_token(mx::array& logits, const SamplingConfig& sampling) {
    // Greedy: argmax on GPU, minimal CPU sync
    if (sampling.temperature <= 0.01f) {
        auto idx = mx::argmax(logits, -1);
        mx::eval(idx);
        return static_cast<uint32_t>(idx.item<uint32_t>());
    }

    // Stochastic sampling on GPU
    float temp = sampling.temperature;
    auto scaled = logits / mx::array(temp);
    auto probs = mx::softmax(scaled, -1);
    // categorical expects 2D [batch, vocab]
    auto probs2d = mx::reshape(probs, {1, -1});
    auto token = mx::random::categorical(probs2d, 1);
    mx::eval(token);
    return static_cast<uint32_t>(token.item<uint32_t>());
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
    // Use bfloat16 weights/activations for faster memory-bound inference.
    model_.promote_for_inference(mx::bfloat16);
}

std::string InferenceEngine::generate(const std::string& prompt, int max_tokens,
                                      const SamplingConfig& sampling) {
    auto tokens = encode_text(prompt, tokenizer_path_);
    if (tokens.empty()) tokens.push_back(static_cast<uint32_t>(model_.config().eos_id));

    int V = model_.config().vocab_size;

    bool quantized = model_.is_quantized();

    // Quantized path uses a growing KV cache with the optimized causal attention.
    // Non-quantized path uses a pre-allocated stable cache and a custom mask.
    std::vector<std::pair<mx::array, mx::array>> kv_cache_growing;
    if (quantized) {
        kv_cache_growing.reserve(model_.config().n_layer);
        for (int i = 0; i < model_.config().n_layer; i++) {
            kv_cache_growing.emplace_back(
                mx::array({0.0f}, mx::float32),
                mx::array({0.0f}, mx::float32));
        }
    }

    mx::array k_cache = mx::zeros({1, model_.config().n_head, block_size_, model_.config().d_model / model_.config().n_head}, mx::bfloat16);
    mx::array v_cache = mx::zeros({1, model_.config().n_head, block_size_, model_.config().d_model / model_.config().n_head}, mx::bfloat16);

    // Prefill
    int T = static_cast<int>(tokens.size());
    if (T > block_size_) {
        tokens.erase(tokens.begin(), tokens.end() - block_size_);
        T = block_size_;
    }
    mx::array idx(mx::array(tokens.data(), {1, T}, mx::uint32));
    mx::array logits = quantized
        ? model_.forward_decode_growing(idx, kv_cache_growing, 0)
        : model_.forward_cached(idx, k_cache, v_cache, 0);
    mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, V}), {V});
    uint32_t next = sample_token(last, sampling);
    tokens.push_back(next);
    int cached_len = T;

    // Decode
    for (int step = 1; step < max_tokens; ++step) {
        if (static_cast<int>(next) == model_.config().eos_id) break;
        if (cached_len >= block_size_) {
            int start = static_cast<int>(tokens.size()) - block_size_;
            std::vector<uint32_t> window(tokens.begin() + start, tokens.end());
            T = block_size_;
            mx::array idx2(mx::array(window.data(), {1, T}, mx::uint32));
            if (quantized) {
                kv_cache_growing.clear();
                for (int i = 0; i < model_.config().n_layer; i++) {
                    kv_cache_growing.emplace_back(
                        mx::array({0.0f}, mx::float32),
                        mx::array({0.0f}, mx::float32));
                }
                logits = model_.forward_decode_growing(idx2, kv_cache_growing, 0);
            } else {
                k_cache = mx::zeros({1, model_.config().n_head, block_size_, model_.config().d_model / model_.config().n_head}, mx::bfloat16);
                v_cache = mx::zeros({1, model_.config().n_head, block_size_, model_.config().d_model / model_.config().n_head}, mx::bfloat16);
                logits = model_.forward_cached(idx2, k_cache, v_cache, 0);
            }
            cached_len = T;
        } else {
            mx::array one(mx::array(&next, {1, 1}, mx::uint32));
            logits = quantized
                ? model_.forward_decode_growing(one, kv_cache_growing, cached_len)
                : model_.forward_cached(one, k_cache, v_cache, cached_len);
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

    int V = model_.config().vocab_size;

    bool quantized = model_.is_quantized();

    std::vector<std::pair<mx::array, mx::array>> kv_cache_growing;
    if (quantized) {
        kv_cache_growing.reserve(model_.config().n_layer);
        for (int i = 0; i < model_.config().n_layer; i++) {
            kv_cache_growing.emplace_back(
                mx::array({0.0f}, mx::float32),
                mx::array({0.0f}, mx::float32));
        }
    }

    mx::array k_cache = mx::zeros({1, model_.config().n_head, block_size_, model_.config().d_model / model_.config().n_head}, mx::bfloat16);
    mx::array v_cache = mx::zeros({1, model_.config().n_head, block_size_, model_.config().d_model / model_.config().n_head}, mx::bfloat16);

    using clock = std::chrono::steady_clock;

    // Prefill
    auto t0 = clock::now();
    int T = static_cast<int>(tokens.size());
    if (T > block_size_) {
        tokens.erase(tokens.begin(), tokens.end() - block_size_);
        T = block_size_;
    }
    mx::array idx(mx::array(tokens.data(), {1, T}, mx::uint32));
    mx::array logits = quantized
        ? model_.forward_decode_growing(idx, kv_cache_growing, 0)
        : model_.forward_cached(idx, k_cache, v_cache, 0);
    mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, V}), {V});
    uint32_t next = sample_token(last, sampling);
    tokens.push_back(next);
    int cached_len = T;
    auto t1 = clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int generated = 1;
    bool stopped = (static_cast<int>(next) == model_.config().eos_id);

    while (!stopped && generated < max_tokens) {
        if (cached_len >= block_size_) {
            int start = static_cast<int>(tokens.size()) - block_size_;
            std::vector<uint32_t> window(tokens.begin() + start, tokens.end());
            T = block_size_;
            mx::array idx2(mx::array(window.data(), {1, T}, mx::uint32));
            if (quantized) {
                kv_cache_growing.clear();
                for (int i = 0; i < model_.config().n_layer; i++) {
                    kv_cache_growing.emplace_back(
                        mx::array({0.0f}, mx::float32),
                        mx::array({0.0f}, mx::float32));
                }
                logits = model_.forward_decode_growing(idx2, kv_cache_growing, 0);
            } else {
                k_cache = mx::zeros({1, model_.config().n_head, block_size_, model_.config().d_model / model_.config().n_head}, mx::bfloat16);
                v_cache = mx::zeros({1, model_.config().n_head, block_size_, model_.config().d_model / model_.config().n_head}, mx::bfloat16);
                logits = model_.forward_cached(idx2, k_cache, v_cache, 0);
            }
            cached_len = T;
        } else {
            mx::array one(mx::array(&next, {1, 1}, mx::uint32));
            logits = quantized
                ? model_.forward_decode_growing(one, kv_cache_growing, cached_len)
                : model_.forward_cached(one, k_cache, v_cache, cached_len);
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
