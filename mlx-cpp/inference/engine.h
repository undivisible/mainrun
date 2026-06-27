#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "model.h"

// ponytail: checkpoint save/load via safetensors
void save_checkpoint(const std::string& path, GPT& model);
void load_checkpoint(const std::string& path, GPT& model);

// ponytail: call Python for tokenization, avoids reimplementing BPE in C++
std::vector<uint32_t> encode_text(const std::string& text, const std::string& tokenizer_path);
std::string decode_ids(const std::vector<uint32_t>& ids, const std::string& tokenizer_path);

struct SamplingConfig {
    float temperature = 0.8f;
};

struct BenchmarkResult {
    int prompt_tokens;
    int tokens_generated;
    double prefill_time_ms;
    double total_time_seconds;
    double tokens_per_second;
};

class InferenceEngine {
public:
    InferenceEngine(const std::string& checkpoint_path,
                    const std::string& tokenizer_path,
                    const GPTConfig& config);

    std::string generate(const std::string& prompt, int max_tokens, const SamplingConfig& sampling);
    BenchmarkResult benchmark(const std::string& prompt, int max_tokens, const SamplingConfig& sampling);
    void quantize() {
        // Quantize in float32 for best quantized-matmul throughput.
        model_.promote_for_inference(mlx::core::float32);
        model_.quantize_for_inference();
    }

private:
    GPT model_;
    std::string tokenizer_path_;
    int block_size_;

    uint32_t sample_token(mlx::core::array& logits, const SamplingConfig& sampling);
};
