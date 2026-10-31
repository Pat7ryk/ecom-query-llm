#!/bin/bash
# 通用 GPU 作业：在 L40S 节点上运行一个 Python 模块（D1–D5，DEVLOG S26）。
#   MODULE=ecomq.eval.mmlu ARGS="--model ... --output ..." sbatch --export=ALL scripts/slurm/gpu_module.sbatch
# ARGS 含空格，必须从环境继承（--export=ALL），不能写进 --export= 列表（见 P12）。
# ENV_NAME 默认 ecomq；D4 的延迟测试用单独的 vllm 环境（ENV_NAME=vllm）。时限用 sbatch --time 覆盖。
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# L40S 只在 TC2N07、TC2N08 上（GRES 未标显卡型号，无法用 --gres=gpu:l40s 指定），排除其余节点
#SBATCH --exclude=TC2N0[1-6]
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --job-name=ecomq-module
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${MODULE:?}"
set -a; source .env; set +a
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-ecomq}"
module load cuda/12.8.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
START=$SECONDS
# shellcheck disable=SC2086
PYTHONPATH=src python -m "$MODULE" ${ARGS:-}
echo "$MODULE: $((SECONDS - START))s"
