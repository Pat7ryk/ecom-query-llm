# N1 实验结果说明

本目录记录 N1 实验：在尽量保持 E2a 训练与评估设置不变的前提下，将基座模型从 **Qwen2.5-7B-Instruct** 替换为 **Qwen2.5-14B-Instruct**，检验扩大模型规模能否稳定提升 Amazon ESCI 四分类任务的表现。

## 当前结论

N1 的三个正式种子均已完成训练、tune 偏置选择和 test 评估：

| Seed | Test raw Macro F1 | Test prior-adjusted Macro F1 |
| ---: | ---: | ---: |
| 42 | 0.542281 | 0.564195 |
| 43 | 0.523131 | 0.564639 |
| 44 | 0.554430 | 0.561737 |
| **Mean** | **0.539947** | **0.563523** |

Prior-adjusted Macro F1 的三种子范围为 **0.561737–0.564639**，三个种子的输出格式有效率均为 100%。结果初步显示 14B 版本表现较好且种子间较稳定。

但是，**目前不能把“14B 稳定优于 7B”作为最终正式结论**。仓库中已有的 E2a 结果使用了旧评估设置，而 N1 使用 NF4、bf16 compute 和 double quantization。必须先用完全相同的协议重新评估三个 E2a adapter，再进行公平比较。

## 实验设置与实际偏差

- 代码 commit：`b6db723e7547a769385e297318b4de4214573628`
- 模型：`Qwen/Qwen2.5-14B-Instruct`
- 模型 commit：`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`
- 训练种子：42、43、44
- 训练轮数：1 epoch，每个种子 625 optimization steps
- 有效 batch size：32
- 评估集：6,000 条 tune、10,000 条 test
- 量化评估：bitsandbytes 4-bit NF4、bf16 compute、double quantization enabled
- 实际 GPU：NVIDIA GeForce RTX 5090 32 GB

原计划使用 A100 80 GB，但实际服务器为 RTX 5090 32 GB。为适配显存，micro-batch 调整为 1，gradient accumulation 调整为 32，有效 batch size 仍保持 32。这个硬件和 micro-batch 偏差需要在最终报告中披露。

三次训练合计记录约 **7.06 GPU-hours**。训练与评估均正常完成，日志中未发现 OOM、NaN、Traceback 或失败运行。

## 目录内容

| 文件 | 用途 |
| --- | --- |
| [`RESULTS.md`](RESULTS.md) | 完整实验摘要、训练记录、指标解释和结论边界 |
| [`training_summary.csv`](training_summary.csv) | 便于查看和导入表格的三种子训练摘要 |
| [`training_summary.json`](training_summary.json) | 机器可读的三种子训练摘要 |
| [`n1-seed42-prior.json`](n1-seed42-prior.json) | Seed 42 的完整指标、per-class P/R/F1、混淆矩阵和偏置曲线 |
| [`n1-seed43-prior.json`](n1-seed43-prior.json) | Seed 43 的完整指标、per-class P/R/F1、混淆矩阵和偏置曲线 |
| [`n1-seed44-prior.json`](n1-seed44-prior.json) | Seed 44 的完整指标、per-class P/R/F1、混淆矩阵和偏置曲线 |
| [`data-manifest.json`](data-manifest.json) | 数据规模、标签分布和 SHA256 provenance |
| [`model-revision.json`](model-revision.json) | Hugging Face 模型版本 provenance |
| [`TEMPLATE.md`](TEMPLATE.md) | 最初预注册的 N1 报告结构与判定标准 |

逐样本 label scores、预测缓存、完整训练日志和 adapter 权重没有提交到 Git：这些文件体积较大或可由现有脚本重新生成。原始结果归档应由实验负责人另行保留。

## 与旧 E2a 结果的关系

旧 E2a 三种子 prior-adjusted Macro F1 为：

- Seed 42：0.549287
- Seed 43：0.553800
- Seed 44：0.540541

旧 E2a 范围为 0.540541–0.553800，N1 范围为 0.561737–0.564639，表面上没有重叠。但这组对比只能作为方向性参考，因为两者的量化评估协议尚未统一。

## 完成正式比较还需要什么

请组员协助提供以下三个 E2a adapter：

1. `e2a-1ep-best`
2. `e5-20k-s43`
3. `e5-20k-s44`

拿到 adapter 后，应按以下顺序完成：

1. 使用 `scripts/autodl/rescore_e2a_for_n1.sh`，以 NF4、bf16 compute、double quantization 统一重算 E2a。
2. 比较 N1 和 E2a 的三种子均值、范围及每类 P/R/F1。
3. 完成 full / clean / merged 分析，并注明 clean labels 的独立来源。
4. 完成 query-level bootstrap 95% CI 和 ranking 指标分析。
5. 补充 N0 gate 或团队批准 waiver，以及 N1 smoke-test 的原始证据。

只有在统一协议后 N1 三个种子仍稳定优于 E2a、种子范围不重叠且 bootstrap CI 支持提升时，才能在最终报告中写成“14B 带来稳定提升”。

## 快速复核

```bash
# 查看核心汇总
cat reports/n1/training_summary.csv

# 提取三个种子的 test 指标
for report in reports/n1/n1-seed*-prior.json; do
  jq '{raw: .test_argmax_no_bias.macro_f1, adjusted: .test_adjusted.macro_f1}' "$report"
done

# 验证所有 JSON 均可解析
for file in reports/n1/*.json; do
  jq empty "$file" || exit 1
done
```

如需引用结果，请优先引用 `RESULTS.md` 中的实验身份和结论边界，不要只复制单个最高 seed 的结果。
