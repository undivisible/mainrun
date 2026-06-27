#include "data.h"
#include "training/trainer.h"
#include "inference/engine.h"
#include "model.h"
#include <cstdio>
#include <string>

static int parse_int(const std::string& s, const std::string& name, int lo, int hi) {
  try {
    int v = std::stoi(s);
    if (v < lo || v > hi) throw std::runtime_error(name + " out of range");
    return v;
  } catch (const std::exception& e) {
    throw std::runtime_error(name + ": invalid value '" + s + "' (" + e.what() + ")");
  }
}

static float parse_float(const std::string& s, const std::string& name, float lo, float hi) {
  try {
    float v = std::stof(s);
    if (v < lo || v > hi) throw std::runtime_error(name + " out of range");
    return v;
  } catch (const std::exception& e) {
    throw std::runtime_error(name + ": invalid value '" + s + "' (" + e.what() + ")");
  }
}

static std::string parse_string(const std::string& s, const std::string& name, size_t max_len) {
  if (s.size() > max_len) throw std::runtime_error(name + ": value too long");
  if (s.find('\0') != std::string::npos) throw std::runtime_error(name + ": contains null byte");
  return s;
}

// ponytail: subcommand + flag parsing, no cxxopts.
int main(int argc, char** argv) {
  std::string data_path = "../data/tokenized_data_24k.json";
  std::string tok_path = "../data/tokenizer_24k.json";
  std::string subcommand = "train";

  int i = 1;
  if (argc > 1 && argv[1] != nullptr && argv[1][0] != '-') {
    subcommand = argv[1];
    i = 2;
  }

  TrainConfig cfg;
  std::string checkpoint = "checkpoint.safetensors";
  std::string prompt = "The future of AI is";
  int max_tokens = 100;
  float temperature = 0.8f;
  bool do_benchmark = false;
  bool use_quantization = false;

  try {
    for (; i < argc; i++) {
      std::string a = argv[i];
      auto next = [&](const std::string& def) -> std::string {
        if (i + 1 < argc) return argv[++i];
        return def;
      };
      if (a == "--max-steps") cfg.max_steps = parse_int(next("889"), "--max-steps", 1, 1000000);
      else if (a == "--max-lr") cfg.max_lr = parse_float(next("0.02"), "--max-lr", 0.0f, 1.0f);
      else if (a == "--warmup-pct") cfg.warmup_pct = parse_float(next("0.05"), "--warmup-pct", 0.0f, 1.0f);
      else if (a == "--decay-pct") cfg.decay_pct = parse_float(next("0.20"), "--decay-pct", 0.0f, 1.0f);
      else if (a == "--eval-interval") cfg.eval_interval = parse_int(next("45"), "--eval-interval", 1, 1000000);
      else if (a == "--batch-size") cfg.batch_size = parse_int(next("32"), "--batch-size", 1, 4096);
      else if (a == "--block-size") cfg.block_size = parse_int(next("256"), "--block-size", 1, 4096);
      else if (a == "--data") data_path = parse_string(next(data_path), "--data", 4096);
      else if (a == "--tokenizer") tok_path = parse_string(next(tok_path), "--tokenizer", 4096);
      else if (a == "--no-ema") cfg.use_ema = false;
      else if (a == "--quick") { cfg.max_steps = 90; cfg.eval_interval = 15; }
      else if (a == "--checkpoint") checkpoint = parse_string(next(checkpoint), "--checkpoint", 4096);
      else if (a == "--prompt") prompt = parse_string(next(prompt), "--prompt", 10000);
      else if (a == "--max-tokens") max_tokens = parse_int(next("100"), "--max-tokens", 1, 1000000);
      else if (a == "--temperature") temperature = parse_float(next("0.8"), "--temperature", 0.0f, 2.0f);
      else if (a == "--benchmark") do_benchmark = true;
      else if (a == "--quantize") use_quantization = true;
      else if (a == "--help" || a == "-h") {
      printf("Usage: %s [subcommand] [options]\n"
             "Subcommands:\n"
             "  train    (default) WSD+Muon+EMA training\n"
             "  infer    Load checkpoint, generate text\n"
             "Options:\n"
             "  --max-steps N      (default 889)\n"
             "  --max-lr F         (default 0.02)\n"
             "  --warmup-pct F     (default 0.05)\n"
             "  --decay-pct F      (default 0.20)\n"
             "  --eval-interval N  (default 45)\n"
             "  --quick            (90 steps, for testing)\n"
             "  --data PATH        (default ../data/tokenized_data_24k.json)\n"
             "  --tokenizer PATH   (default ../data/tokenizer_24k.json)\n"
             "  --no-ema\n"
             "  --checkpoint PATH  (for infer)\n"
             "  --prompt STR       (for infer)\n"
             "  --max-tokens N     (for infer)\n"
             "  --temperature F    (for infer)\n"
             "  --benchmark        (for infer)\n"
             "  --quantize         (for infer, 4-bit weight quantization)\n", argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown arg: %s\n", argv[i]);
      return 1;
    }
  }
  } catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  if (subcommand == "infer") {
    GPTConfig mc;
    DataLoader data(data_path, tok_path, 1, 256);
    mc.vocab_size = data.vocab_size();
    mc.eos_id = data.eos_id();
    InferenceEngine engine(checkpoint, tok_path, mc);
    if (use_quantization) engine.quantize();
    SamplingConfig s{temperature};
    if (do_benchmark) {
      auto r = engine.benchmark(prompt, max_tokens, s);
      printf("Prompt tokens: %d\nGenerated: %d\nPrefill: %.2fms\nTotal: %.2fs\nTokens/sec: %.2f\n",
             r.prompt_tokens, r.tokens_generated, r.prefill_time_ms, r.total_time_seconds, r.tokens_per_second);
    } else {
      printf("Prompt: %s\n", prompt.c_str());
      auto out = engine.generate(prompt, max_tokens, s);
      printf("Output: %s\n", out.c_str());
    }
    return 0;
  }

  DataLoader data(data_path, tok_path, cfg.batch_size, cfg.block_size);

  float best = run_training(data, cfg);
  printf("Best val: %.6f (baseline: %.6f)\n", best, BASELINE);
  if (best < BASELINE) printf("BEAT BASELINE!\n");

  return 0;
}
