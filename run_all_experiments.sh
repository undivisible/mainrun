#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/mlx-cpp"

# Build if needed
if [ ! -f build/train ] || [ model.cpp -nt build/train ] || [ training.cpp -nt build/train ] || [ optimizer.cpp -nt build/train ] || [ inference.cpp -nt build/train ] || [ main.cpp -nt build/train ]; then
  echo "Building..."
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
  cmake --build build -j 2>&1 | tail -3
fi

echo ""
echo "=========================================="
echo "  MLX-CPP Experiment Suite"
echo "  Model: 15M params, 8 layers, Metal GPU"
echo "  Baseline: 0.974555"
echo "=========================================="
echo ""

RESULTS_DIR="../experiment_results"
mkdir -p "$RESULTS_DIR"

# Run each experiment, capture output
run_exp() {
  local name=$1
  shift
  local outfile="$RESULTS_DIR/${name}.log"
  echo "--- Running $name ---"
  time ./build/train "$@" 2>&1 | tee "$outfile"
  echo ""
}

# Individual experiments (945 steps each, ~12 min)
run_exp "baseline"    train --max-steps 945 --eval-interval 45
run_exp "v9_cosine"   v9 --max-steps 945 --eval-interval 45
run_exp "v10_wider"   v10 --max-steps 945 --eval-interval 45
run_exp "v11_deeper"  v11 --max-steps 945 --eval-interval 45
run_exp "v12_low_dropout" v12 --max-steps 945 --eval-interval 45
run_exp "v13_token_aug"   v13 --max-steps 945 --eval-interval 45
run_exp "v14_adaptive_dropout" v14 --max-steps 945 --eval-interval 45

# Summary
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY"
echo "=========================================="
for f in "$RESULTS_DIR"/*.log; do
  name=$(basename "$f" .log)
  best=$(grep "Best val:" "$f" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
  if [ -n "$best" ]; then
    improvement=$(echo "scale=2; (0.974555 - $best) / 0.974555 * 100" | bc)
    status="no_improvement"
    if (( $(echo "$best < 0.974555" | bc -l) )); then
      status="BEAT BASELINE"
    fi
    printf "  %-25s val=%s  improvement=%s%%  %s\n" "$name" "$best" "$improvement" "$status"
  fi
done
echo "=========================================="
