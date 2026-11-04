#!/usr/bin/env bash
# 下载公开数据集原始文件到 data/raw（不进 git）。可重复运行：已下载完整的文件会续传/跳过。
#   bash scripts/download_data.sh            # ESCI + ECInstruct，约 1.7GB
#   ESCI_ONLY=1 bash scripts/download_data.sh # N0/N1 只需要 ESCI
# 在集群头节点运行（下载不属于计算）：RAW_DIR=$DATA_ROOT/raw bash scripts/download_data.sh
# 数据处理属于计算，放在 SLURM 作业里做：scripts/slurm/prepare_data.sbatch
#
# 许可证：
#   ESCI（Shopping Queries Dataset）：Apache-2.0  https://github.com/amazon-science/esci-data
#   ECInstruct：CC-BY-4.0，但各来源子集许可证不同（见其数据卡）  https://huggingface.co/datasets/NingLab/ECInstruct
set -euo pipefail
cd "$(dirname "$0")/.."
RAW="${RAW_DIR:-data/raw}"
mkdir -p "$RAW/esci"

fetch() {  # fetch <url> <dest>
  echo "== $2"
  curl -fL --retry 3 -C - -o "$2" "$1"
}

ESCI=https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset
fetch "$ESCI/shopping_queries_dataset_examples.parquet" "$RAW/esci/shopping_queries_dataset_examples.parquet"
fetch "$ESCI/shopping_queries_dataset_products.parquet" "$RAW/esci/shopping_queries_dataset_products.parquet"
fetch "$ESCI/shopping_queries_dataset_sources.csv"     "$RAW/esci/shopping_queries_dataset_sources.csv"
if [ "${ESCI_ONLY:-0}" != "1" ]; then
  mkdir -p "$RAW/ecinstruct"
  fetch "https://huggingface.co/datasets/NingLab/ECInstruct/resolve/main/dataset.json" "$RAW/ecinstruct/dataset.json"
fi

ls -la "$RAW/esci"
[ "${ESCI_ONLY:-0}" = "1" ] || ls -la "$RAW/ecinstruct"
