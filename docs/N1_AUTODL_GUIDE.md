# N1 AutoDL 单卡工程与运行指南

本文只覆盖 N1（Qwen2.5-14B-Instruct QLoRA）及其必要的 N0 环境门禁，不实现 N2/N3/N4。当前阶段只完成本地代码与检查；没有 GPU 结果。

## 1. 审查结论

### 仓库与协作状态（2026-11-04）

- 最初工作目录不是 Git 仓库，而是 AI6130 Assignment 1 的旧产物。本项目已单独克隆为 `ecom-query-llm`。
- 实验仓库基线：`Xander-YangXH/ecom-query-llm` 的 `main`，审查时为 `81f08d6`；N1 工作位于 `feat/n1-autodl-qwen14b`。
- 最终提交仓库 `Xander-YangXH/NTU-AI6130-LLM-Project` 可访问。其 `main` 只有项目 README；成员通过 `liyouran`、`zhangchen` 远端分支交付结果。
- `liyouran` 分支记录 N0 seed 42 在 L40S 上通过：prior Macro F1 0.5493，落在历史 E2a 三种子 0.5405–0.5538 内。这证明原 CCDS 链路可复现，但不是 AutoDL/A100 环境证明。
- 本地分支不 push、不创建 PR；这些动作必须得到用户后续授权。

### 项目任务与架构

任务输入英文 query 与商品标题，输出 exact / substitute / complement / irrelevant 四分类 JSON。数据由 `prepare_esci` 按 query 抽样，再由 `to_alpaca` 转成 LlamaFactory 数据；训练由 LlamaFactory 做 QLoRA；评估不是自由生成，而是 `score_labels` 对四个完整候选回复求 log probability。`prior_adjust` 只在 tune 上搜索偏置，随后固定应用到 test；`compare`、`noise_aware`、`ranking` 分别提供 query bootstrap、full/clean/merged 与偏置无关排序指标。

### E2a 冻结配置

| 类别 | E2a / N1 正式值 |
| --- | --- |
| 数据 | `e1_relevance` train/val；20,000 / 1,000 条；5,000 / 250 queries |
| 输入与目标 | `qwen` template，cutoff 384，只输出 relevance JSON |
| 量化 | bitsandbytes 4-bit NF4；LlamaFactory v0.9.5 默认 double quantization |
| LoRA | r=16，alpha=32，dropout=0.05，target=all linear modules |
| batch | micro 16 × accumulation 2 = effective 32，单 GPU |
| 优化 | lr 1e-4，cosine，warmup 0.03，1 epoch，bf16 |
| checkpoint | 每 100 步 eval/save；按最低 `eval_loss`；训练结束加载并保存最佳模型 |
| seeds | 42 / 43 / 44（seed 42 为 E2a，43/44 为 E5 的同配置复现） |

N1 唯一研究变量是基座改为 `Qwen/Qwen2.5-14B-Instruct`。`configs/sft/n1_qwen14b_qlora.yaml` 还显式写出 E2a 当时依赖默认值的 bnb、NF4、double quantization 与 gradient checkpointing；这不改变 E2a 语义，只防止依赖升级漂移。

### 数据身份与泄漏边界

| 名称 | 来源 | 用途 | 是否参与选择 |
| --- | --- | --- | --- |
| train | 官方 ESCI train，seed 42/43 的分组抽样 | 梯度更新 | 是 |
| val | 官方 ESCI train，先抽 250 queries | `eval_loss` checkpoint 选择 | 是 |
| tune | 官方 ESCI train，1,500 queries；排除 E1 train/val 与 E2b train | 每个模型单独搜索 prior bias | 是，但只选择推理偏置 |
| test2500 | 官方 ESCI test，2,500 queries；剔除与官方 train 共享的唯一 query | 最终报告 | 否 |

`scripts/autodl/prepare_n1_data.sh` 复用原代码生成这些集合；`ecomq.data.audit_n1` 检查四个集合的 query 两两不重叠、ID 唯一、行数/标签统计，并记录原始与处理后文件 SHA256。仓库没有提交 CCDS 的 canonical hash，因此首次 AutoDL manifest 只能作为新基线；若组长能提供 CCDS 文件，应逐项比对后再训练。

### 直接复用与替换

直接复用：`prepare_esci`、`to_alpaca`、schema/prompt、`score_labels`、`prior_adjust`、`compare`、`noise_aware`、`ranking`、`metrics`、E2a 配置语义和现有单测。

替换：SLURM/PBS 的 `train.sbatch`、`label_scores.sbatch`、`run_e2/e3/e5.sh` 与 module/conda 初始化。AutoDL 使用 `scripts/autodl/` 下的单机脚本，不调用调度器。

### 风险与公平性

1. A100 80GB 很可能容纳 micro 16 的 14B QLoRA，但这仍需真实 smoke 验证。OOM 时脚本停止；只允许把 micro/accum 改为乘积仍为 32的组合，并要求写理由。micro-batch 改变会影响数值与吞吐，不能称为完全等价。
2. 不允许通过缩短 cutoff、减少数据或 epoch 解决正式实验 OOM。smoke 的 `max_samples` 与短 epoch 仅用于工程验证，输出目录与 formal 隔离。
3. bf16 必须由 `torch.cuda.is_bf16_supported()` 实测通过。
4. LlamaFactory v0.9.5 自动从输出目录最新 checkpoint 续训；完成标记存在时脚本拒绝覆盖。
5. 原 `score_labels --load-4bit` 明确使用 NF4 + bf16 compute，但没有显式启用 double quantization；这是 E2a 已发布评估协议的一部分。N1 为公平比较继续复用它。若团队决定改成 double quantization，必须同时重打 E2a 三个种子的分数，不能只改 N1。
6. clean 口径必须使用不参与 E2a/N1 比较的委员会预测；`analyze_n1.sh` 因此要求显式提供 `COMMITTEE_PREDS`，不会用 N1 自己定义争议集。
7. AutoDL 镜像可能带与 CCDS 锁定环境不同的 PyTorch/CUDA。安装脚本不猜版本、不重装 torch，而是先检查镜像，再安装其余锁定核心依赖并保存 freeze。

## 2. GPU 与实例选择

首选单卡 NVIDIA A100 80GB。选择时确认：显存是 80GB（不是 40GB）、实例带持久化数据盘、系统盘空间足够安装 venv、PyTorch 镜像能看到 CUDA。不要选择 24GB/32GB 卡作为正式 N1；A100 40GB 也不适合作为保持 micro 16 的首选。H100/H800 80GB 可作为性能更高的替代，但硬件变化应记录，且 N0 门禁仍要通过。

## 3. Mac → GitHub → AutoDL

本地完成审查、代码和测试后，只在用户授权时 push 功能分支。AutoDL 从相同 commit 拉取；模型、数据、checkpoint 始终在 `$PERSIST_ROOT`，不进 Git。

```bash
git clone https://github.com/Xander-YangXH/ecom-query-llm.git
cd ecom-query-llm
git switch feat/n1-autodl-qwen14b   # 分支 push 后；否则使用用户提供的 commit/patch
cp .env.autodl.example .env
# 如 AutoDL 的持久化挂载点不同，只编辑 .env
bash scripts/autodl/setup_autodl.sh
```

预期：GPU 检查报告显示 A100 80GB、CUDA available=true、bf16_supported=true；依赖 freeze 写到持久化输出目录。

## 4. 数据与模型

```bash
source .venv/bin/activate
bash scripts/autodl/prepare_n1_data.sh
bash scripts/autodl/download_n1_model.sh
```

检查：

```bash
python -m ecomq.data.audit_n1 \
  --data-root "$DATA_ROOT" \
  --output "$DATA_ROOT/n1_data_manifest.recheck.json"
diff "$DATA_ROOT/n1_data_manifest.json" "$DATA_ROOT/n1_data_manifest.recheck.json"
cat "$MODEL_ROOT/Qwen2.5-14B-Instruct/model_revision.json"
```

模型脚本解析并固定 Hugging Face commit，检查 config、tokenizer 和 index 中列出的全部 safetensors shards。若团队指定模型 revision，可在首次下载前设置 `QWEN14B_REVISION=<commit>`。

## 5. 冒烟、N0 门禁与正式训练

先用正式 micro-batch 做短 smoke；它用于显存和 CLI 检查，不是实验结果：

```bash
bash scripts/autodl/train_n1.sh 42 smoke
```

N0 已由组员在 L40S 上通过，但 AutoDL 软件栈与 GPU 不同。严谨做法是在 AutoDL 用原 E2a 配置重跑 seed 42，并完成 tune/test 打分；prior Macro F1 应落在 0.541–0.554：

```bash
bash scripts/download_model.sh Qwen/Qwen2.5-7B-Instruct
bash scripts/autodl/run_n0_gate.sh
test -f "$OUTPUT_ROOT/n0/analysis/N0_GATE_PASSED"
```

该步骤需要 GPU，当前没有伪造结果。若团队书面接受 L40S N0 作为跨环境门禁，至少仍需 N1 smoke、数据 hash 与版本快照全部通过。

正式串行三种子：

```bash
nohup bash scripts/autodl/run_n1_all.sh \
  > "$OUTPUT_ROOT/n1/run-all.log" 2>&1 &
echo $! > "$OUTPUT_ROOT/n1/run-all.pid"
tail -f "$OUTPUT_ROOT/n1/run-all.log"
```

每个 seed 独立目录，完成后有 `COMPLETED`、`run_manifest.json`、训练日志、环境版本和 GPU 显存 CSV。失败时保留 checkpoint；修复原因后用同一命令安全恢复。已完成目录不会被覆盖。

如果 formal micro 16 OOM，先保留日志并停止。经团队确认后，例如改成 2×16：

```bash
export N1_MICRO_BATCH=2
export N1_GRAD_ACCUM=16
export N1_BATCH_OVERRIDE_REASON='A100 80GB 上 micro 16 OOM；保持 effective batch 32'
bash scripts/autodl/train_n1.sh 42 formal
```

不要自动尝试多个 batch，也不要在三个 seed 之间使用不同 batch。

## 6. 正式分析

把 E2a 三个种子的原始 score 与 prior prediction 放到持久化目录。文件名需要与脚本列出的 E2a 名称一致；clean 委员会沿用 S18/S20 的独立 rich-input 三种子预测。

```bash
export E2A_SCORE_DIR="$OUTPUT_ROOT/baseline/e2a/scores"
export E2A_PRED_DIR="$OUTPUT_ROOT/baseline/e2a/preds"
export COMMITTEE_PREDS="$OUTPUT_ROOT/committee/rich-s42.jsonl,$OUTPUT_ROOT/committee/rich-s43.jsonl,$OUTPUT_ROOT/committee/rich-s44.jsonl"
bash scripts/autodl/analyze_n1.sh
```

产出：每 seed 的 prior report（包含 Accuracy、Macro F1、各类 P/R/F1、混淆矩阵）、raw/prior predictions、两组种子的 mean/range 与 query bootstrap、full/clean/merged、排序 AUC/AP。所有 prior bias 只由 tune 得出，再应用到 test。

将真实数值填入 `reports/n1/TEMPLATE.md`，并把小型 JSON 报告复制到 Git 跟踪的 `reports/n1/`；不要提交 scores、数据、模型或 checkpoint。

## 7. 当前完成门槛

本地工程已可进入 AutoDL 部署，但尚不能宣称“可以启动正式 N1”直到以下 GPU 门禁通过：A100/bf16 检查、数据 hash 审核、模型 revision 固定、N1 smoke、以及团队接受的 N0 环境对齐。正式训练、标签打分、统计结果、成本与最终结论全部必须等待真实 GPU。
