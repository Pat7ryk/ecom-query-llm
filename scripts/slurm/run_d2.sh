#!/usr/bin/env bash
# D2（DEVLOG S26）：LoRA 超参消融。与 E2a 只差一项：
#   r8   ：lora_rank=8,  lora_alpha=16（alpha/r 保持 2，与 E2a 的 16/32 相同）
#   r64  ：lora_rank=64, lora_alpha=128
#   qv   ：lora_target=q_proj,v_proj（只训练注意力的 q、v 投影；r=16 不变）
# 每个设定 3 个种子；先跑完种子 42 的三个设定，尽早看到方向。每个模型对调参集和测试集做标签打分。
#   setsid nohup bash scripts/slurm/run_d2.sh >> logs/chain_d2.log 2>&1 < /dev/null &
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MODEL=Qwen2.5-7B-Instruct
CFG=configs/sft/e2_relevance_qlora_1ep_best.yaml
TUNE="$DATA_ROOT/esci_tune/splits/tune.jsonl"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
log() { echo "$(date +%T) [d2] $*"; }
wait_job() {
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}
score() {
  local J S; J=$(sbatch --parsable --export=ALL,MODEL=$MODEL,RUN=$1,TEST=$2,NAME=$3,LOAD_4BIT=1 scripts/slurm/label_scores.sbatch)
  S=$(wait_job "$J"); log "scores $3 job=$J state=$S"
}
declare -A OVR=(
  [r8]="lora_rank=8 lora_alpha=16"
  [r64]="lora_rank=64 lora_alpha=128"
  [qv]="lora_target=q_proj,v_proj"
)
for seed in 42 43 44; do
  for v in r8 r64 qv; do
    RUN=d2-$v-s$seed
    if [ -s "$OUTPUT_ROOT/label_scores/$RUN-test.jsonl" ]; then log "$RUN already scored, skip"; continue; fi
    export TRAIN_OVERRIDES="seed=$seed ${OVR[$v]}"
    RUN=$RUN CONFIG=$CFG DATASET_DIR="$DATA_ROOT/e1_relevance/llamafactory" EVALS=none bash scripts/slurm/chain.sh \
      || { log "train $RUN failed"; unset TRAIN_OVERRIDES; continue; }
    unset TRAIN_OVERRIDES
    score $RUN "$TUNE" $RUN-tune
    score $RUN "$TEST" $RUN-test
  done
done
log "d2 all done"
