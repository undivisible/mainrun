#include "model.h"
#include <mlx/mlx.h>
#include <chrono>
#include <cstdio>

using namespace mlx::core;

int main() {
    GPTConfig cfg;
    cfg.vocab_size = 24000;
    cfg.dropout = 0.0f;

    GPT model(cfg);
    int B = 32, T = 256;
    array idx = random::randint(0, cfg.vocab_size, {B, T}, uint32);
    eval(idx);

    auto forward_fn = std::function<std::vector<array>(const std::vector<array>&)>(
        [&](const std::vector<array>& inputs) -> std::vector<array> {
            auto logits = model.forward(inputs[0], false);
            return {logits};
        });
    auto compiled_forward = compile(forward_fn);

    for (int i = 0; i < 5; i++) {
        eval(compiled_forward({idx})[0]);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; i++) {
        eval(compiled_forward({idx})[0]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 50.0;
    printf("C++ forward only: %.1fms\n", ms);
    return 0;
}
