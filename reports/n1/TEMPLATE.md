# N1：Qwen2.5-14B-Instruct QLoRA 实验报告

> 状态：等待 AutoDL GPU 实验。不得把占位符替换为估算或模拟数据。

## 1. 研究问题与预注册判定标准

- 研究问题：仅把 E2a 的基座从 Qwen2.5-7B-Instruct 换成 Qwen2.5-14B-Instruct，是否带来稳定提升？
- 主要指标：tune 上确定先验偏置后，test 上的 Macro F1。
- 稳定提升：N1 三个种子的先验调整后 Macro F1 均稳定优于 E2a，且两组种子范围不重叠。
- 辅助证据：full / clean / merged、query-level bootstrap 95% CI、raw 指标、排序指标、训练与推理成本。
- 反例处理：若范围重叠、CI 包含 0，或提升只存在于某一个种子/噪声口径，则不报告“稳定提升”。

## 2. 实验身份

| 项目 | 记录值 |
| --- | --- |
| Git commit | `<待填>` |
| 模型 repo / commit | `<待填>` |
| 数据 manifest SHA256 | `<待填>` |
| LlamaFactory | `<待填>` |
| PyTorch / CUDA / driver | `<待填>` |
| GPU | `<待填>` |

## 3. 冻结配置与公平性偏差

- 数据、prompt、标签、训练目标、cutoff、LoRA、学习率、epoch、有效 batch、评估与 checkpoint 选择均继承 E2a。
- 唯一研究变量：7B → 14B。
- 若 micro-batch 从 16 调整：`<理由、OOM 证据、micro/accum、有效 batch、对优化行为的影响>`。
- 其他偏差：`<无 / 逐项说明>`。

## 4. 三种子训练记录

| seed | 最佳 checkpoint | eval_loss | 纯训练耗时 | 峰值显存 | 状态 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 43 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 44 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |

## 5. 主结果

| 系统 | raw Accuracy | raw Macro F1 | prior Accuracy | prior Macro F1 [种子范围] |
| --- | ---: | ---: | ---: | ---: |
| E2a 7B | `<真实值>` | `<真实值>` | `<真实值>` | `<真实值>` |
| N1 14B | `<待 GPU>` | `<待 GPU>` | `<待 GPU>` | `<待 GPU>` |

每个 seed 的 per-class P/R/F1 与 confusion matrix 引用：`<prior report paths>`。

## 6. 稳健性分析

- full / clean / merged：`<引用 n1-vs-e2a-noise-aware.json 并解释>`
- query bootstrap：`<引用 n1-vs-e2a-bootstrap.json>`
- 排序指标（C/I AUC、AP；E-vs-S AUC；relevant-vs-CI）：`<引用 n1-vs-e2a-ranking.json>`
- 是否仅为先验变化：`<分析>`

## 7. 成本与结论

- 训练 GPU 小时：`<待填>`
- 标签打分吞吐与耗时：`<待填>`
- 相比 7B 的成本倍数：`<待填>`
- 最终结论：`<稳定提升 / 未达到稳定提升 / 结果不完整>`
- 局限：标注噪声、单 GPU 环境差异、种子数、量化推理协议等。
