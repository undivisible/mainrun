#include "data.h"
#include "training.h"
#include "inference.h"
#include "model.h"
#include <cstdio>
#include <string>

// ponytail: subcommand + flag parsing, no cxxopts.
int main(int argc, char** argv) {
  std::string data_path = "../tokenized_data_24k.json";
  std::string tok_path = "../tokenizer_24k.json";
  std::string subcommand = "train";

  int i = 1;
  if (argc > 1 && argv[1][0] != '-') {
    subcommand = argv[1];
    i = 2;
  }

  TrainConfig cfg;
  std::string checkpoint = "checkpoint.safetensors";
  std::string prompt = "The future of AI is";
  int max_tokens = 100;
  float temperature = 0.8f;
  int top_k = 50;
  float top_p = 0.9f;
  bool do_benchmark = false;

  for (; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const std::string& def) -> std::string {
      if (i + 1 < argc) return argv[++i];
      return def;
    };
    if (a == "--max-steps") cfg.max_steps = std::stoi(next("945"));
    else if (a == "--max-lr") cfg.max_lr = std::stof(next("0.02"));
    else if (a == "--warmup-pct") cfg.warmup_pct = std::stof(next("0.05"));
    else if (a == "--decay-pct") cfg.decay_pct = std::stof(next("0.20"));
    else if (a == "--eval-interval") cfg.eval_interval = std::stoi(next("45"));
    else if (a == "--batch-size") cfg.batch_size = std::stoi(next("32"));
    else if (a == "--block-size") cfg.block_size = std::stoi(next("256"));
    else if (a == "--data") data_path = next(data_path);
    else if (a == "--tokenizer") tok_path = next(tok_path);
    else if (a == "--no-ema") cfg.use_ema = false;
    else if (a == "--quick") { cfg.max_steps = 90; cfg.eval_interval = 15; }
    else if (a == "--checkpoint") checkpoint = next(checkpoint);
    else if (a == "--prompt") prompt = next(prompt);
    else if (a == "--max-tokens") max_tokens = std::stoi(next("100"));
    else if (a == "--temperature") temperature = std::stof(next("0.8"));
    else if (a == "--top-k") top_k = std::stoi(next("50"));
    else if (a == "--top-p") top_p = std::stof(next("0.9"));
    else if (a == "--benchmark") do_benchmark = true;
    else if (a == "--help" || a == "-h") {
      printf("Usage: %s [subcommand] [options]\n"
             "Subcommands:\n"
             "  train    (default) WSD+Muon+EMA training\n"
             "  all      Run all experiments v9-v14\n"
             "  v9       Cosine LR schedule\n"
             "  v10      Wider model (d_model=768, 6 layers)\n"
             "  v11      Deeper model (d_model=512, 10 layers)\n"
             "  v12      Lower dropout (0.10)\n"
             "  v13      Token augmentation (baseline config)\n"
             "  v14      Adaptive dropout (0.10)\n"
             "  infer    Load checkpoint, generate text\n"
             "Options:\n"
             "  --max-steps N      (default 945)\n"
             "  --max-lr F         (default 0.02)\n"
             "  --warmup-pct F     (default 0.05)\n"
             "  --decay-pct F      (default 0.20)\n"
             "  --eval-interval N  (default 45)\n"
             "  --quick            (90 steps, for testing)\n"
             "  --data PATH        (default ../tokenized_data_24k.json)\n"
             "  --tokenizer PATH   (default ../tokenizer_24k.json)\n"
             "  --no-ema\n"
             "  --checkpoint PATH  (for infer)\n"
             "  --prompt STR       (for infer)\n"
             "  --max-tokens N     (for infer)\n"
             "  --temperature F    (for infer)\n"
             "  --top-k N          (for infer)\n"
             "  --top-p F          (for infer)\n"
             "  --benchmark        (for infer)\n", argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown arg: %s\n", argv[i]);
      return 1;
    }
  }

  if (subcommand == "infer") {
    GPTConfig mc;
    DataLoader data(data_path, tok_path, 1, 256);
    mc.vocab_size = data.vocab_size();
    mc.eos_id = data.eos_id();
    InferenceEngine engine(checkpoint, tok_path, mc);
    SamplingConfig s{temperature, top_k, top_p};
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

  auto run_exp = [&](TrainConfig c) {
    if (cfg.max_steps != 945) { c.max_steps = cfg.max_steps; c.eval_interval = cfg.eval_interval; }
    return run_training(data, c);
  };

  if (subcommand == "all") {
    run_all_experiments(data);
  } else if (subcommand == "v9") run_exp(v9_config());
  else if (subcommand == "v10") run_exp(v10_config());
  else if (subcommand == "v11") run_exp(v11_config());
  else if (subcommand == "v12") run_exp(v12_config());
  else if (subcommand == "v13") run_exp(v13_config());
  else if (subcommand == "v14") run_exp(v14_config());
  else {
    float best = run_training(data, cfg);
    printf("Best val: %.6f (baseline: %.6f)\n", best, BASELINE);
    if (best < BASELINE) printf("BEAT BASELINE!\n");
  }

  return 0;
}
