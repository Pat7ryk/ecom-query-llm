#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source "${ENV_FILE:-.env}"; set +a
: "${OUTPUT_ROOT:?}"
N1_ROOT="${N1_OUTPUT_DIR:-$OUTPUT_ROOT/n1}"
[ -f "$N1_ROOT/smoke-seed42/COMPLETED" ] \
  && [ -s "$N1_ROOT/smoke-seed42/adapter_config.json" ] \
  && [ -s "$N1_ROOT/smoke-seed42/adapter_model.safetensors" ] \
  || { echo "N1 runner blocked: seed 42 smoke is missing or incomplete" >&2; exit 5; }
if [ ! -f "$OUTPUT_ROOT/n0/analysis/N0_GATE_PASSED" ] && [ -z "${N0_GATE_WAIVER_REASON:-}" ]; then
  echo "N1 runner blocked: N0 gate is missing and no team-approved waiver was recorded" >&2
  exit 5
fi
for seed in 42 43 44; do
  if [ -f "$N1_ROOT/formal-seed${seed}/COMPLETED" ]; then
    [ -s "$N1_ROOT/formal-seed${seed}/adapter_config.json" ] \
      && [ -s "$N1_ROOT/formal-seed${seed}/adapter_model.safetensors" ] \
      && [ -s "$N1_ROOT/formal-seed${seed}/run_manifest.json" ] \
      || { echo "seed $seed has a stale/broken COMPLETED marker" >&2; exit 4; }
    echo "seed $seed training already complete; skip"
  else
    bash scripts/autodl/train_n1.sh "$seed" formal
  fi
  for part in tune test; do
    SCORE="$N1_ROOT/label_scores/n1-seed${seed}-${part}.jsonl"
    if [ -s "$SCORE" ] && [ -s "$SCORE.protocol" ]; then
      echo "seed $seed $part scores already complete; skip"
    elif [ -s "$SCORE" ]; then
      echo "existing score file lacks a double-quant protocol marker: $SCORE" >&2
      exit 4
    else
      bash scripts/autodl/score_n1.sh "$seed" "$part"
    fi
  done
done
python - "$N1_ROOT" <<'PY'
import csv, json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for seed in (42, 43, 44):
    run = root / f"formal-seed{seed}"
    manifest = json.loads((run / "run_manifest.json").read_text())
    peak = 0
    for line in (run / "logs/gpu-memory.csv").read_text().splitlines()[1:]:
        match = re.search(r",\s*([0-9]+) MiB,", line)
        if match:
            peak = max(peak, int(match.group(1)))
    if peak <= 0:
        raise SystemExit(f"missing GPU memory samples for seed {seed}")
    rows.append({**manifest, "status": "completed", "peak_gpu_memory_mib": peak})
(root / "training_summary.json").write_text(json.dumps(rows, indent=2))
with (root / "training_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
PY
echo "All N1 seeds and label scores completed. Run analyze_n1.sh next."
