#!/usr/bin/env bash
# Build per-seed prior-adjusted reports, group bootstrap, full/clean/merged,
# and bias-invariant ranking metrics. Baseline inputs are never regenerated here.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source "${ENV_FILE:-.env}"; set +a
source .venv/bin/activate
: "${DATA_ROOT:?}"; : "${OUTPUT_ROOT:?}"
: "${E2A_SCORE_DIR:?set to directory containing e2a/e5 seed tune/test scores}"
: "${E2A_PRED_DIR:?set to directory containing E2a prior-adjusted prediction files}"
: "${COMMITTEE_PREDS:?set to comma-separated committee prediction files for clean metric}"
[ -f "$E2A_SCORE_DIR/../DOUBLE_QUANT_CONFIRMED" ] \
  || { echo "E2a inputs were not confirmed under the N1 double-quant protocol" >&2; exit 2; }

N1_ROOT="${N1_OUTPUT_DIR:-$OUTPUT_ROOT/n1}"
SCORES="$N1_ROOT/label_scores"
ANALYSIS="$N1_ROOT/analysis"
TUNE="$DATA_ROOT/esci_tune/splits/tune.jsonl"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
mkdir -p "$ANALYSIS/preds" "$ANALYSIS/reports"

N1_PRIOR_PREDS=()
N1_RAW_PREDS=()
N1_SCORES=()
for seed in 42 43 44; do
  PREFIX="n1-seed${seed}"
  [ -s "$SCORES/$PREFIX-tune.jsonl.protocol" ] && [ -s "$SCORES/$PREFIX-test.jsonl.protocol" ] \
    || { echo "N1 seed $seed scores lack double-quant protocol markers" >&2; exit 2; }
  python -m ecomq.eval.prior_adjust \
    --tune-recs "$TUNE" --tune-scores "$SCORES/$PREFIX-tune.jsonl" \
    --test-recs "$TEST" --test-scores "$SCORES/$PREFIX-test.jsonl" \
    --report "$ANALYSIS/reports/$PREFIX-prior.json" \
    --preds-out "$ANALYSIS/preds/$PREFIX-prior.jsonl" \
    --raw-preds-out "$ANALYSIS/preds/$PREFIX-raw.jsonl"
  N1_PRIOR_PREDS+=("n1p$seed=$ANALYSIS/preds/$PREFIX-prior.jsonl")
  N1_RAW_PREDS+=("n1r$seed=$ANALYSIS/preds/$PREFIX-raw.jsonl")
  N1_SCORES+=("$SCORES/$PREFIX-test.jsonl")
done

E2A_PRIOR_PREDS=(
  "e2ap42=$E2A_PRED_DIR/e2a-prior.jsonl"
  "e2ap43=$E2A_PRED_DIR/e5-20k-s43-prior.jsonl"
  "e2ap44=$E2A_PRED_DIR/e5-20k-s44-prior.jsonl"
)
E2A_RAW_PREDS=(
  "e2ar42=$E2A_PRED_DIR/e2a-raw.jsonl"
  "e2ar43=$E2A_PRED_DIR/e5-20k-s43-raw.jsonl"
  "e2ar44=$E2A_PRED_DIR/e5-20k-s44-raw.jsonl"
)
python -m ecomq.eval.compare --test "$TEST" \
  --preds "${E2A_RAW_PREDS[@]}" "${E2A_PRIOR_PREDS[@]}" "${N1_RAW_PREDS[@]}" "${N1_PRIOR_PREDS[@]}" \
  --sets 'E2a_raw=e2ar42,e2ar43,e2ar44' 'E2a_prior=e2ap42,e2ap43,e2ap44' \
    'N1_raw=n1r42,n1r43,n1r44' 'N1_prior=n1p42,n1p43,n1p44' \
  --n-boot 1000 --seed 0 --output "$ANALYSIS/reports/n1-vs-e2a-bootstrap.json"

IFS=',' read -r -a COMMITTEE <<< "$COMMITTEE_PREDS"
python -m ecomq.eval.noise_aware --test "$TEST" --committee "${COMMITTEE[@]}" \
  --systems "${E2A_RAW_PREDS[@]}" "${E2A_PRIOR_PREDS[@]}" "${N1_RAW_PREDS[@]}" "${N1_PRIOR_PREDS[@]}" \
  --sets 'E2a_raw=e2ar42,e2ar43,e2ar44' 'E2a_prior=e2ap42,e2ap43,e2ap44' \
    'N1_raw=n1r42,n1r43,n1r44' 'N1_prior=n1p42,n1p43,n1p44' \
  --pairs E2a_raw:N1_raw E2a_prior:N1_prior --n-boot 1000 --seed 0 \
  --output "$ANALYSIS/reports/n1-vs-e2a-noise-aware.json"

python -m ecomq.eval.ranking --test "$TEST" \
  --groups "E2a=$E2A_SCORE_DIR/e2a-test.jsonl,$E2A_SCORE_DIR/e5-20k-s43-test.jsonl,$E2A_SCORE_DIR/e5-20k-s44-test.jsonl" \
  "N1=$(IFS=,; echo "${N1_SCORES[*]}")" \
  --output "$ANALYSIS/reports/n1-vs-e2a-ranking.json"
echo "N1 analysis complete: $ANALYSIS/reports"
