# ecom-query-llm

电商搜索相关性判断的大模型微调：在 Amazon ESCI 数据集上，用 QLoRA 在单卡 L40S 上微调 Qwen2.5-7B-Instruct，判断 query 与商品之间的关系（exact / substitute / complement / irrelevant 四分类），并系统地检查每一项改进在统计上和标注噪声面前是否站得住。

每一步的动机、做法、结果和遇到的问题都记录在 [docs/DEVLOG.md](docs/DEVLOG.md)（S0–S24，问题索引 P1–P18）。

## 任务

- 数据：[Shopping Queries Dataset (ESCI)](https://github.com/amazon-science/esci-data)，美国英文、`large_version=1`。按 query 分组抽样，训练集 5,000 个 query（2 万条），测试集 2,500 个 query（1 万条），另有 1,500 个 query 的调参集；三者没有共同的 query（官方 train/test 之间唯一共享的 query 也已剔除，见 P13）。
- 输入：query 与商品标题。输出：一个 JSON，例如 `{"relevance": "complement"}`。指令、标签定义与解析规则见 [src/ecomq/schema.py](src/ecomq/schema.py)。
- 主指标：宏 F1（exact 约占 67%，只看准确率会被它主导）；同时报告准确率。所有差异都用按 query 分组的 bootstrap 给出 95% 置信区间，关键结论至少 3 个随机种子。

## 主要结果

2,500 个 query 的测试集：

| 方案 | 准确率 | 宏 F1 | 说明 |
|---|---|---|---|
| 基座零样本 | 0.550 | 0.384 | |
| E1：QLoRA，2 个 epoch | 0.727 | 0.534 | |
| **E2a：1 个 epoch，按验证损失选 checkpoint** | **0.739** | 0.535 | 默认训练设定，训练时间减半 |
| E2b：训练时重采样 C/I | 0.721 | 0.545 | 少数类召回升、精确率降 |
| **E2a + 推理时先验调整** | 0.730 | **0.549** | 推荐方案：4 个标签的对数概率加类别偏置，偏置只在调参集上搜索；宏 F1 +1.45（显著），不用重训 |

其余实验（数据量曲线、丰富商品输入、追加 C/I 真实样本、软标签、只追加 complement、先推理再分类）和它们的结论见 DEVLOG 的 S13–S24。简要地说：

- **加数据、加商品文本带来的准确率提升，几乎都发生在标注有争议的样本上**；换成“去掉争议样本”（clean）和“E/S 合并”（merged）两种口径后基本消失（S18）。
- **改训练集的类别比例主要是在改变先验倾向**，与推理时调偏置大部分可以互相替代；只加 complement 会伤害 irrelevant，C/I 各加 3,000 会伤害 E/S 区分（S19、S20）。C/I 各追加 1,500 条（E10）是唯一让两类的排序质量同时提高、E/S 不受损的设定（complement AP 0.322 → 0.347，irrelevant 0.473 → 0.480），但加先验调整后的宏 F1（0.551）仍在 E2a 的种子范围内（S24）。
- **用 7B 模型自己写的理由做“先推理再分类”的 SFT 明显有害**（宏 F1 0.363），因此没有继续做 GRPO（S21–S23）。
- **D1–D5（S26）**：微调后 MMLU 没有净变化（0.735 → 0.735）；LoRA 默认设定（r=16，全部线性层）合适，只训练 q、v 稳定变差 3.2 个百分点；蒸馏到 3B（人工标签）宏 F1 0.517，配合 vLLM 与“只看第一个不同 token”的打分每秒 399 条，是教师现行打分的 47 倍；中文 Multi-CPR 重排 MRR@10 从零样本 0.256 提高到 0.346；意图 / 属性 / 改写多任务在 7B 自标注下没有带来提升。
- **标注审计**：E↔S 之间约 6.8% 的测试样本标注有争议（S17）；complement / irrelevant 方面，盲评估计约 4% 的测试样本粗粒度标注（相关 / 搭配 / 无关）有误，其中标注为 irrelevant 的样本约四分之一更像是相关的（S24）。两项都是 Claude 的盲评，人工复核表在 `analysis/`。

## 方法要点

- **训练**：LLaMA-Factory 0.9.5，QLoRA（4bit NF4，r=16，alpha=32，全部线性层，可训练参数 0.53%），只对回复计损失；训练与推理的提示词由同一个函数生成，并有单测锁定（P1）。
- **评测时打分而不是生成**：对 4 个候选回复分别计算 log p，取最大即预测；同一份打分支持先验调整、召回–精确率曲线和与偏置无关的排序指标（[src/ecomq/eval/score_labels.py](src/ecomq/eval/score_labels.py)）。QLoRA 适配器必须配 4bit 基座评测（P8）。
- **推理时先验调整**：`argmax(log p + b)`，`b_exact = 0`，其余三类的偏置在调参集上用坐标上升最大化宏 F1（[src/ecomq/eval/prior_adjust.py](src/ecomq/eval/prior_adjust.py)）。
- **噪声感知的评测**：clean 口径的争议集由**不参与比较的模型**组成的委员会定义，避免把被比较模型自己的共同错误从考卷上删掉（[src/ecomq/eval/noise_aware.py](src/ecomq/eval/noise_aware.py)）。
- **与偏置无关的排序指标**：每一类直接用 log p(c) 排序计算 AUC / AP；softmax 后的概率并不与偏置无关（P15，[src/ecomq/eval/ranking.py](src/ecomq/eval/ranking.py)）。
- **事先写下判断标准**：从 E8 起，每个实验在运行前把“什么结果算成功”写进 DEVLOG。

## 目录

```
configs/sft/          LLaMA-Factory 配置：e2_relevance_qlora_1ep_best.yaml 为默认（E2a），e9a_* 为推理格式
configs/ds/           DeepSpeed ZeRO-2 / ZeRO-3（NSCC 多卡用）
src/ecomq/schema.py   标签、指令、输出格式与解析、校验
src/ecomq/data/       prepare_esci（抽样与划分）、enrich_esci（商品附加字段）、build_e7 / build_e9（各实验的数据）、
                      make_rationales（推理理由）、to_alpaca（转 LLaMA-Factory 格式）
src/ecomq/eval/       score_labels（标签打分）、score_labels_cot、prior_adjust、compare（bootstrap）、
                      noise_aware、ranking、metrics、generate_hf
src/ecomq/analysis/   es_audit（标注审计：抽样、生成 Excel 盲评表、评分）
scripts/slurm/        CCDS 作业脚本与各实验的调度脚本（run_e*.sh）
scripts/pbs/          NSCC 作业脚本（占位，尚未适配，见 docs/NSCC_PLAN.md）
scripts/              环境安装、模型与数据下载、代码同步（remote.sh）、本地分析缓存（local_cache.sh）、提交前检查（check_leaks.sh）
analysis/             es_audit（E↔S）、ci_audit（complement / irrelevant）两份盲评材料
reports/              各实验的指标与对比 JSON（小文件，进 git）
docs/                 DEVLOG（技术日志）、LABELS（标签说明）、WORKFLOW（工作流）、NSCC_PLAN（下一阶段计划）、TEAM_TODO（组员待办与 NSCC 上手）
tests/                单测（57 个）
```

数据、模型和训练产出只放在服务器上，路径通过 `.env` 配置（见 `.env.ccds.example`），不进 git。

## 复现

本地：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test
```

CCDS（SLURM）上的主结果，按顺序在登录节点用 `setsid nohup` 运行调度脚本（它们只提交和等待作业，不在登录节点计算）：

| 步骤 | 脚本 | 产出 |
|---|---|---|
| 环境、模型、数据 | `scripts/setup_env_ccds.sh`、`scripts/download_model.sh`、`scripts/download_data.sh` | conda 环境、`$MODEL_ROOT`、`$DATA_ROOT/raw/esci` |
| E1 数据 | `sbatch scripts/slurm/prepare_data.sbatch`（`prepare_esci` 默认参数） | `$DATA_ROOT/e1_relevance`（E2a 也用这份数据） |
| E1 训练与评测 | `scripts/slurm/chain_e1.sh` | 适配器 `e1-qlora-r16` |
| E2a / E2b | `scripts/slurm/run_e2.sh` | 适配器 `e2a-1ep-best`、`e2b-rebal-1ep-best` |
| 2,500 query 测试集 | `scripts/slurm/run_r1.sh` | `$DATA_ROOT/esci_test2500` |
| 调参集与先验调整 | `scripts/slurm/run_e3.sh` | `$DATA_ROOT/esci_tune`、标签打分 `e2a-{tune,test}` |

本地分析：`REMOTE_HOST=<user>@<host> bash scripts/local_cache.sh e2a` 会拉取打分并重新生成先验调整的预测与报告（应与 `reports/` 中已提交的报告一致）；之后用 `ecomq.eval.compare`、`noise_aware`、`ranking` 做对比。每个实验的具体命令见 DEVLOG 对应章节。

## 下一阶段

CCDS 单卡上能回答的问题已基本做完。下一阶段在 NSCC 多卡上检验模型容量（14B）与训练方式（全参数微调）是不是瓶颈，计划与需要先核对的事项见 [docs/NSCC_PLAN.md](docs/NSCC_PLAN.md)。

## 数据说明

只使用公开的 ESCI 数据，不含任何真实线上日志。理由数据（E9）由开源模型生成，只用于实验。
