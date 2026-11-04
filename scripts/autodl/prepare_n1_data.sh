#!/usr/bin/env bash
# Reproduce E2a train/val, the original E3 tune exclusion, and test2500.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source "${ENV_FILE:-.env}"; set +a
source .venv/bin/activate
: "${DATA_ROOT:?}"

ESCI_ONLY=1 RAW_DIR="$DATA_ROOT/raw" bash scripts/download_data.sh
python -m ecomq.data.prepare_esci --out-dir "$DATA_ROOT/e1_relevance"
python -m ecomq.data.prepare_esci --out-dir "$DATA_ROOT/e2_rebal" \
  --rebalance complement=0.10,irrelevant=0.20
cmp "$DATA_ROOT/e1_relevance/splits/val.jsonl" "$DATA_ROOT/e2_rebal/splits/val.jsonl"
python -m ecomq.data.prepare_esci --out-dir "$DATA_ROOT/esci_test2500" \
  --test-queries 2500 --only-test
python -m ecomq.data.prepare_esci --out-dir "$DATA_ROOT/esci_tune" \
  --tune-queries 1500 --only-tune \
  --tune-exclude "$DATA_ROOT/e2_rebal/splits/train.jsonl"
python -m ecomq.data.audit_n1 --data-root "$DATA_ROOT" \
  --output "$DATA_ROOT/n1_data_manifest.json"
echo "N1 data ready; immutable manifest: $DATA_ROOT/n1_data_manifest.json"
