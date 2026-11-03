#!/usr/bin/env bash
# 通用实验调度：[可选] 数据准备 → 训练（超时自动续训）→ 评测。在头节点运行，只调用 squeue/sacct/sbatch。
# CCDS 的 QoS 每人最多同时提交 2 个作业，所以逐个提交、等前一个结束（见 DEVLOG P10）。
# 用 setsid nohup 后台运行并写日志，避免本机 SSH 断开时中断（见 DEVLOG P11）：
#   setsid nohup env RUN=... CONFIG=... DATASET_DIR=... bash scripts/slurm/chain.sh >> logs/chain.log 2>&1 < /dev/null &
# 环境变量：
#   RUN, CONFIG, DATASET_DIR       必填
#   TEST                           测试集，默认 $DATA_ROOT/e1_relevance/splits/test.jsonl
#   PREP_ARGS                      非空则先提交 prepare_data 作业（传给 prepare_esci）
#   EVALS                          评测方式，默认 "4bit"，可为 "4bit bf16"；"none" 表示不做生成式评测
#   TRAIN_OVERRIDES                追加给训练作业的配置覆盖（从环境继承）
#   MODEL                          默认 Qwen2.5-7B-Instruct
#   AFTER / TRAIN_JOB              先等待某作业 / 接管已提交的训练作业
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
source scripts/slurm/sbatch_retry.sh

: "${RUN:?}" "${CONFIG:?}" "${DATASET_DIR:?}"
MODEL="${MODEL:-Qwen2.5-7B-Instruct}"
TEST="${TEST:-$DATA_ROOT/e1_relevance/splits/test.jsonl}"
EVALS="${EVALS:-4bit}"
log() { echo "$(date +%T) [$RUN] $*"; }

wait_job() {  # 等作业离开队列，打印最终状态
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}

if [ -n "${AFTER:-}" ]; then S=$(wait_job "$AFTER"); log "wait $AFTER -> $S"; fi

if [ -n "${PREP_ARGS:-}" ]; then
  # PREP_ARGS 可能含逗号，靠 --export=ALL 从环境继承，不写进 --export= 列表
  J=$(PREP_ARGS="$PREP_ARGS" sbatch_retry --parsable --export=ALL,SKIP_INSPECT=1 scripts/slurm/prepare_data.sbatch)
  S=$(wait_job "$J"); log "prepare job=$J state=$S"
  [ "$S" = "COMPLETED" ] || { log "prepare failed, stop"; exit 1; }
fi

for attempt in 1 2 3; do
  if [ "$attempt" = 1 ] && [ -n "${TRAIN_JOB:-}" ]; then
    J=$TRAIN_JOB
  else
    J=$(sbatch_retry --parsable --export=ALL,CONFIG="$CONFIG",MODEL="$MODEL",RUN="$RUN",DATASET_DIR="$DATASET_DIR" scripts/slurm/train.sbatch)
  fi
  S=$(wait_job "$J"); log "train attempt=$attempt job=$J state=$S"
  [ "$S" = "TIMEOUT" ] || break
done
[ "$S" = "COMPLETED" ] || { log "train failed, stop"; exit 1; }

for mode in $EVALS; do
  [ "$mode" = "none" ] && continue
  EXTRA=""; [ "$mode" = "4bit" ] && EXTRA=",LOAD_4BIT=1"
  J=$(sbatch_retry --parsable --export=ALL,MODEL="$MODEL",RUN="$RUN",LORA=1,TEST="$TEST",BATCH_SIZE=32,MAX_NEW_TOKENS=24$EXTRA scripts/slurm/eval.sbatch)
  S=$(wait_job "$J"); log "eval-$mode job=$J state=$S"
done
log "chain done"
