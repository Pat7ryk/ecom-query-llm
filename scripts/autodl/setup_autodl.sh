#!/usr/bin/env bash
# Create an isolated environment while reusing the CUDA-compatible PyTorch from
# the selected AutoDL image. This script does not guess or replace torch/CUDA.
set -euo pipefail
cd "$(dirname "$0")/../.."

ENV_FILE="${ENV_FILE:-.env}"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE; cp .env.autodl.example .env" >&2; exit 1; }
set -a; source "$ENV_FILE"; set +a
: "${DATA_ROOT:?}"; : "${MODEL_ROOT:?}"; : "${OUTPUT_ROOT:?}"; : "${HF_HOME:?}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys
print("python", sys.version.replace("\n", " "))
if sys.version_info < (3, 11):
    raise SystemExit("LlamaFactory 0.9.5 requires a newer Python image; choose a compatible AutoDL image")
PY

"$PYTHON_BIN" - <<'PY'
try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is absent. Choose an AutoDL PyTorch+CUDA image before running setup") from exc
print("image torch", torch.__version__, "torch CUDA", torch.version.cuda)
PY

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv --system-site-packages .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -e ".[dev]" -r requirements-autodl.txt
mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$OUTPUT_ROOT" "$HF_HOME" logs

python - <<'PY'
import importlib.metadata as md
import torch
for name in ("llamafactory", "transformers", "peft", "bitsandbytes", "accelerate"):
    print(name, md.version(name))
print("torch", torch.__version__, "CUDA", torch.version.cuda, "CUDA available", torch.cuda.is_available())
PY

bash scripts/autodl/gpu_check.sh
python -m pip freeze --exclude-editable > "$OUTPUT_ROOT/environment-freeze.txt"
echo "AutoDL environment ready; version snapshot: $OUTPUT_ROOT/environment-freeze.txt"
