#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f "${ENV_FILE:-.env}" ] && { set -a; source "${ENV_FILE:-.env}"; set +a; }
: "${OUTPUT_ROOT:?source .env first}"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/environment"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="$OUTPUT_ROOT/environment/gpu-check-$STAMP.txt"
{
  uname -a
  nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
  .venv/bin/python - <<'PY'
import sys, torch
print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA")
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("bf16_supported", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    raise SystemExit("bf16 is required by the frozen E2a/N1 protocol")
PY
} | tee "$REPORT"
echo "GPU check passed: $REPORT"
