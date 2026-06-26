#!/bin/bash
# Benchmark: Rust GPT training vs Python baseline (0.974555)
# Runs both and compares final validation loss.

set -e

PYTHON_BASELINE=0.974555
RUST_DIR="./rust-experiments"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

echo "================================================"
echo "  GPT TRAINING BENCHMARK: Rust vs Python"
echo "  Python baseline: $PYTHON_BASELINE"
echo "================================================"
echo ""

# --- Rust ---
echo "[1/2] Running Rust training (945 steps, Muon + EMA)..."
RUST_LOG="$LOG_DIR/rust_benchmark.log"
cd "$RUST_DIR"
cargo run --release -- train --max-steps 945 --max-lr 0.02 --warmup-pct 0.05 --decay-pct 0.20 2>&1 | tee "$RUST_LOG"
RUST_VAL=$(grep "Best EMA val loss" "$RUST_LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
cd ..
echo "Rust final val loss: $RUST_VAL"
echo ""

# --- Python ---
echo "[2/2] Running Python training (7 epochs, Muon + EMA)..."
PYTHON_LOG="$LOG_DIR/python_benchmark.log"
cd mainrun
python train.py 2>&1 | tee "$PYTHON_LOG"
PYTHON_VAL=$(grep "validation_step" "$PYTHON_LOG" | tail -1 | grep -oE '"loss": [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
cd ..
echo "Python final val loss: $PYTHON_VAL"
echo ""

# --- Comparison ---
echo "================================================"
echo "  RESULTS"
echo "================================================"
echo "  Python baseline:  $PYTHON_BASELINE"
echo "  Python (this run): $PYTHON_VAL"
echo "  Rust (this run):   $RUST_VAL"
echo ""
if [ -n "$RUST_VAL" ] && [ -n "$PYTHON_VAL" ]; then
    RUST_DIFF=$(echo "scale=4; ($RUST_VAL - $PYTHON_VAL)" | bc)
    echo "  Rust - Python:     $RUST_DIFF"
    if (( $(echo "$RUST_VAL < $PYTHON_VAL" | bc -l) )); then
        echo "  ✅ Rust BEATS Python!"
    else
        echo "  ❌ Rust does not beat Python yet"
    fi
fi
echo "================================================"
echo "Logs: $RUST_LOG, $PYTHON_LOG"
