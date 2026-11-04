#!/usr/bin/env bash
# Train exactly one N1 seed. Formal defaults preserve E2a batch 16 x accum 2.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
set -a; source "${ENV_FILE:-.env}"; set +a
source .venv/bin/activate

SEED="${1:?usage: train_n1.sh <42|43|44> [formal|smoke]}"
MODE="${2:-formal}"
[[ "$SEED" =~ ^(42|43|44)$ ]] || { echo "seed must be 42, 43, or 44" >&2; exit 2; }
[[ "$MODE" =~ ^(formal|smoke)$ ]] || { echo "mode must be formal or smoke" >&2; exit 2; }
: "${DATA_ROOT:?}"; : "${MODEL_ROOT:?}"; : "${OUTPUT_ROOT:?}"
MODEL="$MODEL_ROOT/Qwen2.5-14B-Instruct"
DATASET_DIR="$DATA_ROOT/e1_relevance/llamafactory"
CONFIG="configs/sft/n1_qwen14b_qlora.yaml"
MICRO="${N1_MICRO_BATCH:-16}"
ACCUM="${N1_GRAD_ACCUM:-2}"

if [ "$MODE" = formal ] && { [ "$MICRO" != 16 ] || [ "$ACCUM" != 2 ]; }; then
  [ "$((MICRO * ACCUM))" -eq 32 ] || { echo "formal effective batch must remain 32" >&2; exit 2; }
  [ -n "${N1_BATCH_OVERRIDE_REASON:-}" ] || { echo "set N1_BATCH_OVERRIDE_REASON for a formal batch override" >&2; exit 2; }
fi

RUN_ROOT="$OUTPUT_ROOT/n1"
RUN="$RUN_ROOT/${MODE}-seed${SEED}"
mkdir -p "$RUN_ROOT" "$RUN/logs"
[ ! -f "$RUN/COMPLETED" ] || { echo "refusing to overwrite completed run: $RUN" >&2; exit 3; }
command -v llamafactory-cli >/dev/null
[ -s "$DATASET_DIR/dataset_info.json" ] || { echo "missing prepared E2a data" >&2; exit 1; }
[ -s "$MODEL/config.json" ] || { echo "missing downloaded 14B model" >&2; exit 1; }

bash scripts/autodl/gpu_check.sh
python -m pip freeze --exclude-editable > "$RUN/logs/pip-freeze.txt"
git rev-parse HEAD > "$RUN/logs/git-commit.txt"
cp "$DATA_ROOT/n1_data_manifest.json" "$RUN/logs/data-manifest.json"
cp "$MODEL/model_revision.json" "$RUN/logs/model-revision.json"

GPU_LOG="$RUN/logs/gpu-memory.csv"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 15 > "$GPU_LOG" &
GPU_LOGGER=$!
cleanup() { kill "$GPU_LOGGER" 2>/dev/null || true; }
trap cleanup EXIT

OVERRIDES=(
  "model_name_or_path=$MODEL"
  "dataset_dir=$DATASET_DIR"
  "output_dir=$RUN"
  "seed=$SEED"
  "data_seed=$SEED"
  "per_device_train_batch_size=$MICRO"
  "gradient_accumulation_steps=$ACCUM"
)
if [ "$MODE" = smoke ]; then
  OVERRIDES+=("max_samples=128" "num_train_epochs=0.02" "eval_strategy=no" \
    "save_strategy=no" "load_best_model_at_end=false" "per_device_eval_batch_size=1")
fi

export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
START="$(date +%s)"
set +e
llamafactory-cli train "$CONFIG" "${OVERRIDES[@]}" 2>&1 | tee "$RUN/logs/train.log"
STATUS=${PIPESTATUS[0]}
set -e
END="$(date +%s)"
if [ "$STATUS" -ne 0 ]; then
  printf '%s\n' "$STATUS" > "$RUN/FAILED_EXIT_CODE"
  echo "training failed with exit $STATUS; output kept for diagnosis/resume" >&2
  exit "$STATUS"
fi
[ -s "$RUN/adapter_config.json" ] || { echo "training returned success but adapter_config.json is missing" >&2; exit 4; }
[ -s "$RUN/adapter_model.safetensors" ] || { echo "training returned success but adapter weights are missing" >&2; exit 4; }

python - "$RUN" "$SEED" "$MODE" "$MICRO" "$ACCUM" "$START" "$END" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
state = run / "trainer_state.json"
trainer = json.loads(state.read_text()) if state.exists() else {}
manifest = {
    "seed": int(sys.argv[2]), "mode": sys.argv[3],
    "micro_batch": int(sys.argv[4]), "gradient_accumulation": int(sys.argv[5]),
    "effective_batch": int(sys.argv[4]) * int(sys.argv[5]),
    "started_epoch_s": int(sys.argv[6]), "ended_epoch_s": int(sys.argv[7]),
    "elapsed_seconds": int(sys.argv[7]) - int(sys.argv[6]),
    "best_model_checkpoint": trainer.get("best_model_checkpoint"),
    "best_metric": trainer.get("best_metric"),
}
(run / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
PY
touch "$RUN/COMPLETED"
echo "completed: $RUN"
