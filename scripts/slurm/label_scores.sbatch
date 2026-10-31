#!/bin/bash
# 计算 relevance 四个候选标签的对数概率（推理时先验调整用，见 DEVLOG S12）。
#   sbatch --export=ALL,MODEL=Qwen2.5-7B-Instruct,RUN=e2a-1ep-best,TEST=<jsonl>,NAME=<结果名>,LOAD_4BIT=1 scripts/slurm/label_scores.sbatch
# 输出：$OUTPUT_ROOT/label_scores/$NAME.jsonl
# RUN 留空：不加载适配器，直接给基座打分（零样本基线）。TASK=relevance_zh：中文二分类（D5）。
# COT=1：relevance_cot 模型（E9a），先生成理由再以理由为条件打分（ecomq.eval.score_labels_cot）；
#   生成较慢，提交时用 --time=03:00:00 覆盖默认时限。
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# L40S 只在 TC2N07、TC2N08 上（GRES 未标显卡型号，无法用 --gres=gpu:l40s 指定），排除其余节点
#SBATCH --exclude=TC2N0[1-6]
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:30:00
#SBATCH --job-name=ecomq-lblscore
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${MODEL:?}"; : "${TEST:?}"; : "${NAME:?}"
set -a; source .env; set +a
module load anaconda
eval "$(conda shell.bash hook)"
conda activate ecomq
module load cuda/12.8.0

ARGS=(--model "$MODEL_ROOT/$MODEL" --test "$TEST"
      --output "$OUTPUT_ROOT/label_scores/$NAME.jsonl" --batch-size "${BATCH_SIZE:-32}")
[ -n "${RUN:-}" ] && ARGS+=(--lora "$OUTPUT_ROOT/$RUN")
[ -n "${TASK:-}" ] && ARGS+=(--task "$TASK")
[ -n "${LOAD_4BIT:-}" ] && ARGS+=(--load-4bit)
START=$SECONDS
if [ -n "${COT:-}" ]; then
  PYTHONPATH=src python -m ecomq.eval.score_labels_cot "${ARGS[@]}" --max-new-tokens "${MAX_NEW_TOKENS:-96}"
else
  PYTHONPATH=src python -m ecomq.eval.score_labels "${ARGS[@]}"
fi
echo "score_labels: $((SECONDS - START))s"
