#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/mainrun"

export MAINRUN_LOCAL=1

PYTHON=""
for c in python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    PYTHON="$c"
    break
  fi
done
[[ -n "$PYTHON" ]] || { echo "no python3 found"; exit 1; }
VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "using: $PYTHON (Python $VER)"
if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -U pip
pip install -q -r "$ROOT/.devcontainer/requirements-train.txt"

echo "=== device check ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
if torch.backends.mps.is_available():
    print("device: mps (Apple GPU)")
elif torch.cuda.is_available():
    print("device: cuda")
else:
    print("device: cpu")
PY

echo "=== download dataset (cached in mainrun/data) ==="
python download_dataset.py

echo "=== train 7 epochs (log: mainrun/logs/mainrun.log) ==="
python train.py

echo "=== best validation loss ==="
python - <<'PY'
import json
from pathlib import Path
p = Path("logs/mainrun.log")
best = None
for line in p.read_text().splitlines():
    o = json.loads(line)
    if o.get("event") == "validation_step":
        best = o["loss"] if best is None else min(best, o["loss"])
print(f"best validation_step loss: {best}")
print("baseline to beat: 1.754")
PY