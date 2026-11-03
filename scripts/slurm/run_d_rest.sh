#!/usr/bin/env bash
# D1、D3、D4、D5（DEVLOG S26）的调度脚本；D2 另有 run_d2.sh，两个调度脚本各自一次只提交一个作业，可以同时运行
# （CCDS 每人最多同时提交 2 个作业）。每一步都检查产出是否已存在，已存在就跳过，所以中断后可以直接重新启动。
#   STAGES="d1 d5 d3 d4" setsid nohup bash scripts/slurm/run_d_rest.sh >> logs/chain_drest.log 2>&1 < /dev/null &
# 头节点上需要先准备好（只下载，不计算）：
#   $DATA_ROOT/raw/mmlu/all/{dev,test}-*.parquet           （cais/mmlu）
#   $DATA_ROOT/raw/multicpr/ecom/*                          （Multi-CPR 电商子集）
#   $MODEL_ROOT/{Qwen2.5-1.5B-Instruct,Qwen2.5-3B-Instruct} （D4 学生）、$MODEL_ROOT/$ANNOTATOR（D3 标注模型）
#   conda 环境 vllm（D4 的延迟测试；装好后用 VLLM=1 启动，否则跳过 vLLM 部分）
set -eo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
source scripts/slurm/sbatch_retry.sh
BASE=Qwen2.5-7B-Instruct
CFG=configs/sft/e2_relevance_qlora_1ep_best.yaml
E1="$DATA_ROOT/e1_relevance/splits"
TUNE="$DATA_ROOT/esci_tune/splits/tune.jsonl"
TEST="$DATA_ROOT/esci_test2500/splits/test.jsonl"
POOL="${POOL:-$DATA_ROOT/esci_train50k/splits/train.jsonl}"   # E4 的 5 万条嵌套训练集（包含 E2a 的 2 万条）
ANNOTATOR="${ANNOTATOR:-Qwen2.5-7B-Instruct}"   # 家目录配额放不下 14B（S26），默认用 7B 基座零样本标注
LS="$OUTPUT_ROOT/label_scores"
STAGES="${STAGES:-d1 d5 d3 d4}"
log() { echo "$(date +%T) [drest] $*"; }
wait_job() {
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  sacct -j "$1" -X -n -o State | head -1 | tr -d ' '
}
# submit <名字> <sbatch 参数...>：提交并等待；失败时返回非零
submit() {
  local name=$1; shift
  local J S; J=$(sbatch_retry --parsable "$@"); S=$(wait_job "$J"); log "$name job=$J state=$S"
  [ "$S" = "COMPLETED" ]
}
module_job() {  # module_job <名字> <时限> <模块> <参数字符串> [ENV_NAME]
  MODULE=$3 ARGS=$4 ENV_NAME=${5:-ecomq} submit "$1" --time="$2" --export=ALL scripts/slurm/gpu_module.sbatch
}
prep_job() {  # prep_job <名字> <模块> <参数字符串>
  PREP_MODULE=$2 PREP_ARGS=$3 submit "$1" --export=ALL,SKIP_INSPECT=1 scripts/slurm/prepare_data.sbatch
}
score() {  # score <名字> <基座> <适配器，可空> <测试集> [额外的 --export 项]
  [ -s "$LS/$1.jsonl" ] && { log "$1 exists, skip"; return 0; }
  submit "scores $1" --time=03:00:00 --export=ALL,MODEL=$2,RUN=$3,TEST=$4,NAME=$1${5:+,$5} scripts/slurm/label_scores.sbatch
}
train() {  # train <RUN> <基座> <配置> <数据目录> [TRAIN_OVERRIDES]
  [ -f "$OUTPUT_ROOT/$1/adapter_config.json" ] && { log "$1 trained, skip"; return 0; }
  # save_only_model=true：checkpoint 不存优化器状态，省一半磁盘，不影响训练结果
  TRAIN_OVERRIDES="save_only_model=true ${5:-}" RUN=$1 MODEL=$2 CONFIG=$3 DATASET_DIR=$4 EVALS=none bash scripts/slurm/chain.sh
}

d1() {  # MMLU：基座与 E2a 的 3 个种子，都用 4bit，与相关性评测时的加载方式一致
  local OUT="$OUTPUT_ROOT/d1_mmlu"; mkdir -p "$OUT"
  local -A RUNS=([base]="" [e2a-s42]=e2a-1ep-best [e2a-s43]=e5-20k-s43 [e2a-s44]=e5-20k-s44)
  for name in base e2a-s42 e2a-s43 e2a-s44; do
    [ -s "$OUT/$name.jsonl" ] && continue
    local lora=""; [ -n "${RUNS[$name]}" ] && lora="--lora $OUTPUT_ROOT/${RUNS[$name]}"
    module_job "mmlu $name" 02:00:00 ecomq.eval.mmlu \
      "run --model $MODEL_ROOT/$BASE $lora --load-4bit --data-dir $DATA_ROOT/raw/mmlu/all --output $OUT/$name.jsonl" || true
  done
}

d5() {  # 中文：Multi-CPR 电商子集，零样本与微调（3 个种子）在 BM25 前 20 名上重排
  local D="$DATA_ROOT/d5_multicpr"
  [ -s "$D/splits/dev_rerank.jsonl" ] || prep_job "d5 prep" ecomq.data.prepare_multicpr \
    "--raw-dir $DATA_ROOT/raw/multicpr/ecom --out-dir $D" || return 1
  local DEV="$D/splits/dev_rerank.jsonl"
  score d5-zeroshot-dev $BASE "" "$DEV" "LOAD_4BIT=1,TASK=relevance_zh"
  for seed in 42 43 44; do
    train d5-mcpr-s$seed $BASE $CFG "$D/llamafactory" "seed=$seed" || { log "train d5-mcpr-s$seed failed"; continue; }
    score d5-mcpr-s$seed-dev $BASE d5-mcpr-s$seed "$DEV" "LOAD_4BIT=1,TASK=relevance_zh"
  done
}

d3() {  # 意图、属性、改写：本地开源模型标注真实 query，多任务（3 个种子）与单任务（1 个种子）
  local A="$OUTPUT_ROOT/d3"; mkdir -p "$A"
  [ -s "$A/annot_trainval.jsonl" ] || module_job "d3 annotate train/val" 04:00:00 ecomq.data.annotate_query \
    "--model $MODEL_ROOT/$ANNOTATOR --inputs $E1/train.jsonl $E1/val.jsonl --output $A/annot_trainval.jsonl" || return 1
  [ -s "$A/annot_test.jsonl" ] || module_job "d3 annotate test" 03:00:00 ecomq.data.annotate_query \
    "--model $MODEL_ROOT/$ANNOTATOR --inputs $TEST --output $A/annot_test.jsonl" || return 1
  local D="$DATA_ROOT/d3_multitask"
  [ -s "$D/stats.json" ] || prep_job "d3 build" ecomq.data.build_d3 \
    "--rel-dir $E1 --full $A/annot_trainval.jsonl --out-dir $D" || return 1
  for seed in 42 43 44; do
    train d3-mt-s$seed $BASE $CFG "$D/mt/llamafactory" "seed=$seed" || { log "train d3-mt-s$seed failed"; continue; }
    score d3-mt-s$seed-tune $BASE d3-mt-s$seed "$TUNE" LOAD_4BIT=1
    score d3-mt-s$seed-test $BASE d3-mt-s$seed "$TEST" LOAD_4BIT=1
  done
  train d3-fullonly-s42 $BASE $CFG "$D/full_only/llamafactory" "seed=42" || log "train d3-fullonly-s42 failed"
  for run in d3-mt-s42 d3-fullonly-s42; do
    [ -s "$A/$run-preds.jsonl" ] || module_job "d3 generate $run" 02:00:00 ecomq.eval.generate_hf \
      "--model $MODEL_ROOT/$BASE --lora $OUTPUT_ROOT/$run --test $A/annot_test.jsonl --output $A/$run-preds.jsonl --load-4bit --batch-size 32 --max-new-tokens 160" || true
  done
}

d4() {  # 蒸馏：教师 E2a（seed 42）+ 先验调整 → 1.5B / 3B 学生；再测部署速度
  score e2a-pool50k $BASE e2a-1ep-best "$POOL" LOAD_4BIT=1 || return 1
  local D="$DATA_ROOT/d4_distill"
  [ -s "$D/stats.json" ] || prep_job "d4 build" ecomq.data.build_d4 \
    "--base $E1/train.jsonl --pool $POOL --pool-scores $LS/e2a-pool50k.jsonl --val $E1/val.jsonl --tune $TUNE --tune-scores $LS/e2a-tune.jsonl --out-dir $D" || return 1
  local SCFG=configs/sft/d4_student_lora.yaml
  local -A DATA=([gold]="$DATA_ROOT/e1_relevance/llamafactory" [t20k]="$D/teacher20k/llamafactory" [t50k]="$D/teacher50k/llamafactory")
  for seed in 42 43 44; do
    for size in 1.5b 3b; do
      local M=Qwen2.5-1.5B-Instruct; [ $size = 3b ] && M=Qwen2.5-3B-Instruct
      for v in gold t50k t20k; do
        [ $v = t20k ] && [ $seed != 42 ] && continue   # teacher20k 只做 1 个种子（机制对照）
        local RUN=d4-$size-$v-s$seed
        train $RUN $M $SCFG "${DATA[$v]}" "seed=$seed" || { log "train $RUN failed"; continue; }
        score $RUN-tune $M $RUN "$TUNE"
        score $RUN-test $M $RUN "$TEST"
      done
    done
  done
  local OUT="$OUTPUT_ROOT/d4_latency"; mkdir -p "$OUT"
  local -A LAT=([teacher-hf4bit]="$BASE e2a-1ep-best --load-4bit" [s3b-hf]="Qwen2.5-3B-Instruct d4-3b-t50k-s42"
                [s1.5b-hf]="Qwen2.5-1.5B-Instruct d4-1.5b-t50k-s42")
  for name in teacher-hf4bit s3b-hf s1.5b-hf; do
    [ -s "$OUT/$name.json" ] && continue
    set -- ${LAT[$name]}
    module_job "latency $name" 01:00:00 ecomq.eval.latency \
      "--engine hf --model $MODEL_ROOT/$1 --lora $OUTPUT_ROOT/$2 ${3:-} --test $TEST --name $name --output $OUT/$name" || true
  done
  if [ "${VLLM:-0}" = 1 ]; then
    for spec in "teacher-vllm $BASE e2a-1ep-best" "s3b-vllm Qwen2.5-3B-Instruct d4-3b-t50k-s42" \
                "s1.5b-vllm Qwen2.5-1.5B-Instruct d4-1.5b-t50k-s42"; do
      set -- $spec
      [ -s "$OUT/$1.json" ] && continue
      module_job "latency $1" 01:00:00 ecomq.eval.latency \
        "--engine vllm --model $MODEL_ROOT/$2 --lora $OUTPUT_ROOT/$3 --test $TEST --name $1 --output $OUT/$1" vllm || true
    done
  else
    log "VLLM!=1, skip vLLM latency"
  fi
}

for stage in $STAGES; do
  log "stage $stage start"
  $stage || log "stage $stage stopped early"
done
log "drest all done"
