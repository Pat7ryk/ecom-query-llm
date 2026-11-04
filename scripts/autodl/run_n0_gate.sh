#!/usr/bin/env bash
# Optional but recommended AutoDL cross-environment gate: reproduce E2a seed 42.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
set -a; source "${ENV_FILE:-.env}"; set +a
source .venv/bin/activate
: "${DATA_ROOT:?}"; : "${MODEL_ROOT:?}"; : "${OUTPUT_ROOT:?}"
MODEL="$MODEL_ROOT/Qwen2.5-7B-Instruct"
DATASET_DIR="$DATA_ROOT/e1_relevance/llamafactory"
RUN="$OUTPUT_ROOT/n0/formal-seed42"
SCORES="$OUTPUT_ROOT/n0/label_scores"
ANALYSIS="$OUTPUT_ROOT/n0/analysis"
mkdir -p "$RUN/logs" "$SCORES" "$ANALYSIS"
[ -s "$MODEL/config.json" ] || { echo "download 7B first: bash scripts/download_model.sh Qwen/Qwen2.5-7B-Instruct" >&2; exit 1; }

if [ ! -f "$RUN/COMPLETED" ]; then
  bash scripts/autodl/gpu_check.sh
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 15 \
    > "$RUN/logs/gpu-memory.csv" &
  GPU_LOGGER=$!
  trap 'kill "$GPU_LOGGER" 2>/dev/null || true' EXIT
  export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
  llamafactory-cli train configs/sft/e2_relevance_qlora_1ep_best.yaml \
    "model_name_or_path=$MODEL" "dataset_dir=$DATASET_DIR" "output_dir=$RUN" \
    seed=42 data_seed=42 2>&1 | tee "$RUN/logs/train.log"
  [ -s "$RUN/adapter_config.json" ] && [ -s "$RUN/adapter_model.safetensors" ]
  touch "$RUN/COMPLETED"
  kill "$GPU_LOGGER" 2>/dev/null || true
  trap - EXIT
fi

for part in tune test; do
  if [ "$part" = tune ]; then
    INPUT="$DATA_ROOT/esci_tune/splits/tune.jsonl"
  else
    INPUT="$DATA_ROOT/esci_test2500/splits/test.jsonl"
  fi
  OUT="$SCORES/n0-$part.jsonl"
  if [ ! -s "$OUT" ]; then
    PYTHONPATH=src python -m ecomq.eval.score_labels --model "$MODEL" --lora "$RUN" \
      --test "$INPUT" --output "$OUT" --batch-size "${N0_SCORE_BATCH:-32}" --load-4bit
  fi
done

python -m ecomq.eval.prior_adjust \
  --tune-recs "$DATA_ROOT/esci_tune/splits/tune.jsonl" --tune-scores "$SCORES/n0-tune.jsonl" \
  --test-recs "$DATA_ROOT/esci_test2500/splits/test.jsonl" --test-scores "$SCORES/n0-test.jsonl" \
  --report "$ANALYSIS/n0-prior.json" --preds-out "$ANALYSIS/n0-prior.jsonl" \
  --raw-preds-out "$ANALYSIS/n0-raw.jsonl"
python - "$ANALYSIS/n0-prior.json" "$ANALYSIS/N0_GATE_PASSED" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
score = report["test_adjusted"]["macro_f1"]
print("N0 prior-adjusted Macro F1:", score)
if not 0.541 <= score <= 0.554:
    raise SystemExit("N0 gate failed: investigate versions, data hashes, batch, checkpoint, and scoring before N1")
Path(sys.argv[2]).write_text(f"macro_f1={score}\n")
PY
echo "N0 gate passed: $ANALYSIS/N0_GATE_PASSED"
