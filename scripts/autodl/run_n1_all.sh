#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source "${ENV_FILE:-.env}"; set +a
: "${OUTPUT_ROOT:?}"
for seed in 42 43 44; do
  if [ -f "$OUTPUT_ROOT/n1/formal-seed${seed}/COMPLETED" ]; then
    echo "seed $seed training already complete; skip"
  else
    bash scripts/autodl/train_n1.sh "$seed" formal
  fi
  for part in tune test; do
    if [ -s "$OUTPUT_ROOT/n1/label_scores/n1-seed${seed}-${part}.jsonl" ]; then
      echo "seed $seed $part scores already complete; skip"
    else
      bash scripts/autodl/score_n1.sh "$seed" "$part"
    fi
  done
done
echo "All N1 seeds and label scores completed. Run analyze_n1.sh next."
