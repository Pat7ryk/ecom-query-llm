#!/usr/bin/env bash
# Reuse the team's exact score_labels protocol: 4-bit NF4 base + adapter.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
set -a; source "${ENV_FILE:-.env}"; set +a
source .venv/bin/activate
SEED="${1:?usage: score_n1.sh <42|43|44> <tune|test>}"
PART="${2:?usage: score_n1.sh <42|43|44> <tune|test>}"
[[ "$SEED" =~ ^(42|43|44)$ ]] || exit 2
[[ "$PART" =~ ^(tune|test)$ ]] || exit 2
: "${DATA_ROOT:?}"; : "${MODEL_ROOT:?}"; : "${OUTPUT_ROOT:?}"
N1_ROOT="${N1_OUTPUT_DIR:-$OUTPUT_ROOT/n1}"
MODEL="${N1_MODEL_PATH:-$MODEL_ROOT/Qwen2.5-14B-Instruct}"
ADAPTER="$N1_ROOT/formal-seed${SEED}"
if [ "$PART" = tune ]; then
  INPUT="${N1_TUNE_PATH:-$DATA_ROOT/esci_tune/splits/tune.jsonl}"
else
  INPUT="${N1_TEST_PATH:-$DATA_ROOT/esci_test2500/splits/test.jsonl}"
fi
OUT_DIR="$N1_ROOT/label_scores"
OUT="$OUT_DIR/n1-seed${SEED}-${PART}.jsonl"
mkdir -p "$OUT_DIR" "$N1_ROOT/logs"
[ -f "$ADAPTER/COMPLETED" ] || { echo "training not complete: $ADAPTER" >&2; exit 1; }
[ ! -s "$OUT" ] || { echo "refusing to overwrite existing score file: $OUT" >&2; exit 3; }

START=$SECONDS
PYTHONPATH=src python -m ecomq.eval.score_labels \
  --model "$MODEL" --lora "$ADAPTER" --test "$INPUT" --output "$OUT" \
  --batch-size "${N1_SCORE_BATCH:-32}" --load-4bit --double-quant \
  2>&1 | tee "$N1_ROOT/logs/score-seed${SEED}-${PART}.log"
printf '%s\n' 'bnb 4-bit NF4, bf16 compute, double quantization enabled' > "$OUT.protocol"
echo "score_labels seed=$SEED part=$PART elapsed=$((SECONDS - START))s output=$OUT"
