#!/usr/bin/env bash
# Re-score all three E2a seeds with the exact N1 inference quantization protocol.
# Required because historical E2a scores used NF4 without double quantization.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source "${ENV_FILE:-.env}"; set +a
source .venv/bin/activate
: "${DATA_ROOT:?}"; : "${MODEL_ROOT:?}"; : "${OUTPUT_ROOT:?}"
: "${E2A_ADAPTER_ROOT:?directory containing e2a-1ep-best and e5-20k-s43/s44 adapters}"
MODEL="$MODEL_ROOT/Qwen2.5-7B-Instruct"
ROOT="$OUTPUT_ROOT/baseline/e2a-double-quant"
SCORES="$ROOT/scores"
PREDS="$ROOT/preds"
REPORTS="$ROOT/reports"
mkdir -p "$SCORES" "$PREDS" "$REPORTS"
[ -s "$MODEL/config.json" ] || { echo "missing Qwen2.5-7B-Instruct" >&2; exit 1; }

for spec in 'e2a:e2a-1ep-best' 'e5-20k-s43:e5-20k-s43' 'e5-20k-s44:e5-20k-s44'; do
  IFS=: read -r name adapter_name <<< "$spec"
  ADAPTER="$E2A_ADAPTER_ROOT/$adapter_name"
  [ -s "$ADAPTER/adapter_config.json" ] || { echo "missing adapter: $ADAPTER" >&2; exit 1; }
  for part in tune test; do
    if [ "$part" = tune ]; then INPUT="$DATA_ROOT/esci_tune/splits/tune.jsonl"; else INPUT="$DATA_ROOT/esci_test2500/splits/test.jsonl"; fi
    OUT="$SCORES/$name-$part.jsonl"
    if [ ! -s "$OUT" ]; then
      PYTHONPATH=src python -m ecomq.eval.score_labels --model "$MODEL" --lora "$ADAPTER" \
        --test "$INPUT" --output "$OUT" --batch-size "${E2A_SCORE_BATCH:-32}" \
        --load-4bit --double-quant
      printf '%s\n' 'bnb 4-bit NF4, bf16 compute, double quantization enabled' > "$OUT.protocol"
    elif [ ! -s "$OUT.protocol" ]; then
      echo "existing E2a score lacks a double-quant protocol marker: $OUT" >&2
      exit 2
    fi
  done
  python -m ecomq.eval.prior_adjust \
    --tune-recs "$DATA_ROOT/esci_tune/splits/tune.jsonl" --tune-scores "$SCORES/$name-tune.jsonl" \
    --test-recs "$DATA_ROOT/esci_test2500/splits/test.jsonl" --test-scores "$SCORES/$name-test.jsonl" \
    --report "$REPORTS/$name-prior.json" --preds-out "$PREDS/$name-prior.jsonl" \
    --raw-preds-out "$PREDS/$name-raw.jsonl"
done
printf '%s\n' 'bnb 4-bit NF4, bf16 compute, double quantization enabled' > "$ROOT/DOUBLE_QUANT_CONFIRMED"
echo "E2a re-scoring complete: $ROOT"
