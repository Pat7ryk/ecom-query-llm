# 电商相关性项目：下一步待办

2026-10-28 · yang

CCDS 单卡阶段已经结束，结果见仓库 `docs/DEVLOG.md` 的 S0–S24。下一步迁移到 NSCC，检验模型容量和训练方式是不是瓶颈。两份标注表的人工复核不需要 GPU，可以同时进行。认领任务时，在表格里填上负责人并更新状态。

## 项目概况

本项目微调一个 7B 大模型，判断电商搜索中 query 与商品是什么关系，并系统地检验每一项改进在统计上、在标注噪声面前是否站得住。

| 方面 | 内容 |
| --- | --- |
| 任务 | 输入 query 与商品标题，输出四个类别之一：exact（完全满足）、substitute（可替代）、complement（可搭配）、irrelevant（不相关）。输出格式是 JSON，例如 `{"relevance": "complement"}` |
| 数据 | Amazon 公开的 ESCI 数据集（美国英文）。训练集 5,000 个 query（2 万条），测试集 2,500 个 query（1 万条），调参集 1,500 个 query（6,000 条），三个集合没有共同的 query |
| 模型与训练 | Qwen2.5-7B-Instruct，QLoRA（4bit 量化，r=16，只训练 0.53% 的参数），框架是 LLaMA-Factory |
| 评测 | 对 4 个标签分别计算对数概率，取最大的作为预测，不让模型自由生成。主指标是宏 F1：exact 约占 67%，只看准确率会被这一类主导。所有差异都用按 query 分组的 bootstrap 给出 95% 置信区间，关键结论跑 3 个随机种子 |
| 两种辅助口径 | clean：去掉标注有争议的 E/S 样本；merged：把 exact 与 substitute 合并成“相关”。用来判断一项提升是不是只出现在标注本身就模糊的样本上 |
| 算力 | 目前所有实验都在 NTU CCDS 集群的单卡 L40S（48GB，SLURM）上完成；下一阶段使用 NSCC 的多卡（PBS） |
| 资料 | 公开的 GitHub 仓库 [ecom-query-llm](https://github.com/Xander-YangXH/ecom-query-llm)：`README.md` 是总览；`docs/DEVLOG.md` 记录每一步的原因、做法、结果和遇到的问题（S0–S25，P1–P18）；`docs/NSCC_PLAN.md` 是下一阶段计划 |

范围说明：最初的提案打算用一个模型同时完成意图识别、属性抽取、Query 改写和相关性判断。前三项没有合适的公开标签，所以目前只做了相关性判断。是否恢复这三项，见文末“待定”一节。

## 已完成的结果

目前推荐的方案是 **E2a 训练 + 推理时先验调整**，宏 F1 0.548（零样本基座为 0.384）。之后的实验都没能在 3 个种子上稳定超过它。

**主要指标**（2,500 个 query 的测试集；标注“3 种子”的行是三个种子的均值，方括号内是最差到最好的种子）

| 方案 | 准确率 | 宏 F1 | 说明 |
| --- | --- | --- | --- |
| 基座零样本 | 0.550 | 0.384 | 未微调 |
| E1：QLoRA，2 个 epoch | 0.727 | 0.534 | |
| E2a：1 个 epoch，按验证损失选 checkpoint | 0.739 | 0.535 | 默认的训练设定，训练时间减半 |
| **E2a + 先验调整（3 种子）** | 0.728 | **0.548** [0.541, 0.554] | 推荐方案：给 4 个标签各加一个偏置，偏置只在调参集上搜索，不需要重训 |
| E8：追加 1,500 条 complement + 先验调整（3 种子） | 0.732 | 0.551 [0.544, 0.556] | 可选的小改进 |
| E10：complement、irrelevant 各追加 1,500 条 + 先验调整（3 种子） | 0.728 | 0.551 [0.548, 0.555] | 排序质量最好的模型 |

**实验演进**（按时间顺序；括号里是技术日志的章节号）

| 实验 | 做法 | 结论 |
| --- | --- | --- |
| E1（S8） | QLoRA 微调 2 个 epoch | 微调远好于零样本，宏 F1 从 0.384 升到 0.534 |
| E2（S10） | 1 个 epoch 并选最优 checkpoint（E2a）；训练时重采样少数类（E2b） | E2a 准确率更高、训练时间减半，成为默认设定；E2b 少数类的召回升高、精确率下降 |
| R1（S11） | 测试集从 600 个 query 扩大到 2,500 个 | 之前的结论在更大的测试集上仍然成立 |
| E3（S12） | 推理时先验调整 | 不用重训，宏 F1 提高 1.45 个百分点（显著），效果与 E2b 相当 |
| E4 / E5（S15） | 训练数据从 2 万条增加到 20 万条；3 个种子 | 准确率在约 10 万条后饱和；不同种子之间可以相差 1.3 个点，所以关键结论必须用多个种子 |
| E6（S16） | 输入中加入品牌、颜色、商品要点 | 准确率提高 0.5 个点，加先验调整后宏 F1 没有变化，计算成本是原来的 1.5 倍，不采用 |
| E↔S 审计（S17） | 抽取“所有模型都判错”的 E/S 样本做盲评 | 约 6.8% 的测试样本标注有争议 |
| 噪声感知评测（S18） | 用 clean、merged 两种口径复核所有结论 | 加数据、加输入带来的提升，几乎都出现在有争议的样本上 |
| E7（S19） | complement、irrelevant 各追加 3,000 条，并加软标签 | 主要改变的是模型的先验倾向；complement 的排序有小幅真实提升；软标签没有效果 |
| E8（S20） | 只追加 complement：1,500 / 3,000 / 6,000 条 | 1,500 到 3,000 条时收益饱和，6,000 条反而有害，还会拖累 irrelevant |
| E9a（S21–S23） | 先写一句理由再给标签的 SFT，原计划之后接 GRPO | 宏 F1 降到 0.363，按事先定的标准停止这个方向 |
| C/I 审计（S24） | 做法同 S17，对象换成 complement / irrelevant | 约 4% 的测试样本粗粒度标注有误；标为 irrelevant 的样本中，约四分之一更像是相关的 |
| E10（S24） | complement、irrelevant 各追加 1,500 条 | 两类的排序质量同时提高，但宏 F1 仍在 E2a 的种子范围之内 |

**目前的结论**

- 稳定的收益只有两项：微调本身，以及推理时先验调整。其余改动的收益都在种子差异之内。
- 调整训练集的类别比例，主要是在改变模型的先验倾向，大部分可以用推理时调偏置代替。
- 在 7B + QLoRA 的条件下，瓶颈主要在标注质量和商品关系知识，而不是训练技巧。模型容量和训练方式是否也是瓶颈，还没有答案，这是下面待办的重点。

## 对接约定

开始任何任务之前，先读 README、`docs/WORKFLOW.md`，以及技术日志开头的问题索引（P1–P18）。

- **先写标准再跑实验**：每个实验运行前，先把“什么结果算成功”写进 `docs/DEVLOG.md`。
- **比较口径**：每个设定跑 3 个种子，与 E2a 的 3 个种子比较。报告 full / clean / merged 三种口径，加与不加先验调整都要报，再加上与偏置无关的排序指标。只有加先验调整后，两组种子的取值范围不重叠，才算稳定提升。
- **记录**：每一步的原因、做法、结果都写进技术日志。遇到的问题按“现象 → 原因 → 解决 → 验证”记录，并登记到问题索引。指标 JSON 放在 `reports/` 下，提交进 git。
- **不进仓库的内容**：数据、模型、checkpoint；服务器主机名、用户名、密码、本机路径。每次提交前运行 `bash scripts/check_leaks.sh`。
- **集群规则**：不在登录节点上跑计算，一律提交作业；只访问自己的家目录或项目目录；调度脚本用 `setsid nohup` 启动。

## NSCC 快速上手

以下内容整理自 NSCC 的官方 [ASPIRE 2A FAQ](https://help.nscc.sg/aspire2a/faqs/) 和 [用户指南](https://help.nscc.sg/aspire2a/user-guide/)，查阅日期为 2026-10-28。规则以官网为准；本组的项目号等信息，在 A1、A2 核实后补到这里。

**1. 注册账号**

- NTU 的学生和教职工不用填申请表：在 [user.nscc.sg](https://user.nscc.sg/) 用 NTU 账号登录，第一次登录就会自动开通 NSCC 账号。以后重置 NSCC 密码、更换 SSH 公钥也在这个页面。
- 不属于 NTU、NUS、A\*STAR 等合作机构的成员，要到 [new.nscc.sg](https://new.nscc.sg/) 申请，NSCC 说明处理需要约 7 个工作日。

**2. 加入项目，拿到项目号**

- 提交作业必须带项目号（`-P <项目号>`）。家目录只有 50 GB，而且不会扩容，放不下 7B 和 14B 模型加数据，所以必须用项目目录。
- 项目在 [NSCC Projects Portal](https://projects.nscc.sg/) 申请，或者由项目负责人把组员加进来。申请说明见 [Call for Projects](https://help.nscc.sg/nscc-call-for-project)。
- 本组的项目号：`<待 A1 确认后填写>`。登录后运行 `myprojects`，可以查看自己所属的项目和剩余额度。
- 计费方式：GPU 作业每 GPU 小时消耗 64 SU。按 `docs/NSCC_PLAN.md` 的估算，N0 到 N3 合计约 40 到 50 GPU 小时，大约是 2,600 到 3,200 SU（估算），申请项目额度时可以参考。

**3. 登录**

NTU 用户的登录节点是 `aspire2antu.nscc.sg`，用 NSCC 的用户名和密码（或 SSH 密钥）登录。从哪里连决定了要不要经过跳板机。以下来自 NTU HPC Users' Group 的 SharePoint 页面 “Using NTU JumpHost to NSCC ASPIRE-2A”，需要 NTU 账号才能打开；跳板机的地址也写在那个页面上，这里不写。

- **在 NTU 校园内**：不需要跳板机，直接 `ssh <NSCC 用户名>@aspire2antu.nscc.sg`。连 NTU-SECURE 无线网是否算在校内，页面上没有答复，第一次登录时试一下即可。
- **在校外**：NTU VPN 不允许直接连到 NSCC，必须分两跳：
  1. 连上 NTU VPN。
  2. 用 NTU 账号和密码 SSH 登录 NTU HPCC 的跳板机。
  3. 在跳板机上用 NSCC 账号 `ssh <NSCC 用户名>@aspire2antu.nscc.sg`。
- **跳板机账号**：要把自己的 NTU ID 登记到 HPCC 跳板机，并关联导师的 NTU ID（用于追溯和费用分摊）。登记方式页面上没有写明，联系 NTU HPC 团队。也可以用自己学院放在校内的电脑或工作站当跳板机，Windows 机器需要开启远程桌面。
- **跳板机的限制**：每人 50 GB；不能存放数据，管理员可能不经通知直接删除；每人同时最多建立 2 个到 NSCC 的连接。大量数据请用校内的电脑中转。
- **NTU VPN**：学生默认没有 VPN 权限，需要由导师（PI）填写 NTU 的 VPN 申请表，通常一周内会收到确认邮件。VPN 入口：教职工用 vpngate-staff.ntu.edu.sg，学生用 vpngate-student.ntu.edu.sg。
- **省事的写法**：在本机 `~/.ssh/config` 里配置一次，之后 `ssh nscc` 就会自动经过跳板机，`rsync` 和 `scp` 同样适用：

```
Host ntu-jump
    HostName <跳板机地址，见 SharePoint 页面>
    User <NTU 用户名>
Host nscc
    HostName aspire2antu.nscc.sg
    User <NSCC 用户名>
    ProxyJump ntu-jump
```

- 非 NTU 的入口：先连 ASPIRE 2A 的 VPN，再登录 `aspire2a.nscc.sg` 或 `login.asp2a.nscc.sg`。NUS、A\*STAR、SUTD 的用户各有自己的登录节点，见 FAQ。
- 建议在 user.nscc.sg 上传 SSH 公钥，用密钥登录 NSCC。

**4. 存储**

| 目录 | 配额 | 清理规则 | 本项目放什么 |
| --- | --- | --- | --- |
| `/home`（家目录） | 50 GB，不扩容 | 不清理 | 代码：`git clone https://github.com/Xander-YangXH/ecom-query-llm.git` |
| `/home/project/<项目号>` | 按项目批准的额度 | 不清理 | 模型、数据、训练产出；在 `.env` 中设置 `DATA_ROOT`、`MODEL_ROOT`、`OUTPUT_ROOT`、`HF_HOME` |
| `/scratch` | 100 TB | 超过 30 天未使用的文件自动删除 | 训练时读写量大的临时文件；结果要拷回项目目录 |

- NSCC 不做备份，重要的结果要放在项目目录，或者提交进 git（`reports/`）。
- 查看用量：`myquota` 查家目录，`myquota -p <项目号>` 查项目目录。
- 从 2025-01-02 起，NSCC 屏蔽了部分外部网站。如果下载模型或数据失败，先下载到本地，再用 `rsync` 或 `scp` 上传。

**5. 环境**

- `cp .env.nscc.example .env`，按上表填写路径。然后 `module load miniforge3`（或其他合适的 Python 模块），再运行 `bash scripts/setup_env.sh`，依赖版本见任务 A3。
- `module avail` 列出可用软件。不要用 `module purge`，需要时用 `module rm` 或 `module swap`。如果作业里报 `module: command not found`，在 `~/.bashrc` 里加一行 `. /etc/bashrc`。

**6. 提交 GPU 作业（PBS）**

- normal 队列会按申请的 GPU 数自动分配到子队列：g1（1 张卡，最长 24 小时）、g2（2 到 3 张）、g3（4 张，即 1 个节点）、g4（5 张以上）、glong（24 到 120 小时）、gdev（1 到 4 张，最长 2 小时，适合调试）。ai 队列需要另外申请。
- 每张 GPU 配 16 个 CPU 核和 124 GB 内存。每人在全系统最多可以有 100 个作业，可以用作业依赖 `-W depend=afterok:` 串联多个步骤（任务 A6）。
- 交互式调试：`qsub -I -l select=1:ngpus=1 -l walltime=02:00:00 -P <项目号> -q normal`
- 批处理作业模板。这是参照 FAQ 改写的，还没有在本项目里跑过；正式版本以任务 A4 改好的 `scripts/pbs/train.pbs` 为准：

```bash
#!/bin/bash
#PBS -N ecomq-train
#PBS -P <项目号>
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o logs/
cd "$PBS_O_WORKDIR" || exit $?
set -a; source .env; set +a
source .venv/bin/activate
llamafactory-cli train configs/sft/e2_relevance_qlora_1ep_best.yaml
```

- 常用命令：`qsub <脚本>` 提交，`qstat -u $USER` 查看，`qdel <作业号>` 取消，`myusage` 查看自己的 GPU 用量。

**7. 求助**

- [ASPIRE 2A 用户指南](https://help.nscc.sg/aspire2a/user-guide/)，包括 Quick Start、PBS 参考卡、AI 作业入门等 PDF
- [FAQ](https://help.nscc.sg/aspire2a/faqs/)
- [服务台](https://keris.service-now.com/csm)
- [系统状态](https://help.nscc.sg/service)

## 第一阶段：NSCC 准备

这一阶段完成之前，所有 NSCC 实验都不能开始。仓库里的 PBS 脚本目前还是占位，细节见 `docs/NSCC_PLAN.md` 第二节。

| 编号 | 任务 | 完成标准 | 依赖 | 负责人 | 状态 |
| --- | --- | --- | --- | --- | --- |
| A0 | 每位组员开通访问权限：NSCC 账号（user.nscc.sg）；在校外工作的，还要开通 NTU VPN（学生由导师申请）和 HPCC 跳板机登记 | 能 `ssh` 到 `aspire2antu.nscc.sg`，`myprojects` 能看到本组项目 | A1（项目号） | 每位组员 | 未开始 |
| A1 | 核对账号与调度：项目号、队列名、`select=` 资源写法、单个作业的时间上限、每人可同时提交的作业数 | 写进 `NSCC_PLAN.md` 的检查表，并补进上方“NSCC 快速上手” | — | | 未开始 |
| A2 | 核对硬件与存储：GPU 型号与显存、项目目录路径与配额、计算节点能否访问外网、Python 与 CUDA 的 module 名称 | 写进检查表；据此确定 N3 的批大小和 ZeRO 设置 | — | | 未开始 |
| A3 | 安装环境：`scripts/setup_env.sh` 按 `requirements-lock-ccds.txt` 固定版本（transformers 5.6.0、peft 0.18.1、trl 0.24.0、LLaMA-Factory 0.9.5、torch 2.11） | 在计算节点上导入成功，能看到 GPU | A2 | | 未开始 |
| A4 | 修改 `scripts/pbs/train.pbs`：从 `DATASET_DIR` 读取数据，支持 `TRAIN_OVERRIDES`（种子等） | 行为与 `train.sbatch` 一致 | A1 | | 未开始 |
| A5 | 修改 `scripts/pbs/eval.pbs`：改为标签打分（`ecomq.eval.score_labels`），测试集换成 `esci_test2500` | 输出格式与 CCDS 上的打分文件相同 | A1 | | 未开始 |
| A6 | 编写 PBS 调度脚本（相当于 `chain.sh`）：提交 → 等待 → 失败重试 → 下一步；条件允许时改用 `-W depend=afterok:` | 冒烟测试的完整流程能无人值守跑通 | A4、A5 | | 未开始 |
| A7 | 准备数据：重新下载 ESCI，用同样的命令和种子生成 `e1_relevance`、`esci_test2500`、`esci_tune` | 每个文件的 `md5sum` 都与 CCDS 上的一致 | A3 | | 未开始 |
| A8 | 下载模型：Qwen2.5-7B-Instruct、Qwen2.5-14B-Instruct | 文件完整，包括 `config.json` 和分词器（见 P4） | A2 | | 未开始 |
| A9 | 多卡冒烟测试：用冒烟数据验证 `FORCE_TORCHRUN=1` 和 DeepSpeed ZeRO-2 / ZeRO-3 | 多卡训练和打分作业都正常结束 | A3、A4、A8 | | 未开始 |

## 第二阶段：NSCC 实验

按 N0 → N2 → N1 → N3 的顺序进行。N0 不通过，后面的所有对比都不可信。N2 最便宜，而且它决定 N1、N3 用量化版还是 bf16 版作对照。每个模型都要对调参集和测试集做标签打分，然后在本地跑先验调整、三种口径和排序指标。

| 编号 | 实验 | 验收标准 | 依赖 | 算力（估算，GPU 小时） | 负责人 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| N0 | 跨集群检查：按 E2a 的设定重训 seed 42 | 加先验调整后的宏 F1 落在 CCDS 三个种子的范围内（0.541 到 0.554）；超出范围时，先排查库版本、量化实现、批大小和数据文件 | A1–A7 | 1.5 | | 未开始 |
| N2 | 7B bf16 LoRA（不量化），其余与 E2a 相同，3 个种子 | 运行前把判断标准写进 DEVLOG；回答 4bit 量化是否损失效果 | N0 | 4 | | 未开始 |
| N1 | Qwen2.5-14B-Instruct QLoRA，E2a 设定，3 个种子 | 同上；回答更大的模型是否更好 | N0、A8 | 9 | | 未开始 |
| N3 | 7B 全参数微调，ZeRO-3，4 张卡；学习率重新搜索（2 到 3 档），每档 3 个种子 | 同上；回答全参数微调是否优于 LoRA | N0、A9 | 25 到 35 | | 未开始 |
| N4（可选） | 用 72B 级模型零样本判断两份盲评样本 | 与 Claude 盲评和人工结果比较一致率 | N0；最好在 B1、B2 之后 | 待定 | | 未开始 |

如果 N1、N3 在三种口径下都没有稳定提升，结论就是：在这份数据上，限制效果的是标注质量和任务本身的歧义，而不是模型或训练方式。

## 可并行：标注人工复核

两次标注审计目前只有 Claude 的盲评，需要至少一位组员做人工判断。这项工作不需要 GPU，随时可以做。标注时不要打开同一目录下的 `key.json` 和 Claude 的结果文件，否则盲评就失效了。

| 编号 | 任务 | 完成标准 | 依赖 | 负责人 | 状态 |
| --- | --- | --- | --- | --- | --- |
| B1 | 标注 E↔S 盲评表 `analysis/es_audit/blind_sheet.xlsx`（120 条，其中 20 条是对照） | 每一条都在下拉框里选了标签 | — | | 未开始 |
| B2 | 标注 C/I 盲评表 `analysis/ci_audit/blind_sheet.xlsx` | 同上 | — | | 未开始 |
| B3 | 用 `ecomq.analysis.es_audit score` 评分（用法见各目录的 `README.md`），计算人工与原标注、人工与 Claude 的一致率 | 结果写进 DEVLOG（补充 S17、S24），并更新 README 和面试文档中的争议比例 | B1、B2 | | 未开始 |

## 仓库与文档

| 编号 | 任务 | 完成标准 | 依赖 | 负责人 | 状态 |
| --- | --- | --- | --- | --- | --- |
| R1 | 清理历史提交中的本地路径（P9、P14），并且只保留本人为贡献者 | 10-28 已完成：改写历史并强制推送，路径已替换，见 DEVLOG S25。之前克隆过仓库的组员需要重新克隆 | — | yang | 已完成 |
| R2 | 让 `scripts/local_cache.sh` 支持 NSCC 的存储路径（目前写死为 CCDS 家目录下的路径） | 能从 NSCC 拉取打分，并重建预测和报告 | A2 | | 未开始 |
| R3 | 每个实验结束后，更新 DEVLOG 的结果、README 的主要结果和面试文档的实验表 | 报告 JSON 存进 `reports/`；每个结论都能追溯到对应的配置和指标文件 | 各实验 | | 未开始 |

## 待定：原计划中暂缓的项

这些是项目提案里有、但还没做的内容。前两项在 CCDS 单卡上就能做，不必等 NSCC；其余的等第二阶段有结论后再决定。

| 编号 | 任务 | 暂缓原因 | 何时决定 | 负责人 | 状态 |
| --- | --- | --- | --- | --- | --- |
| D1 | 通用能力回归抽测：MMLU 5-shot，比较 E2a 与基座 | 已完成：MMLU 基座 0.7349，E2a 三个种子 0.7353–0.7355，没有净变化，达到标准（DEVLOG S26 结果一） | 随时可做（CCDS） | | 已完成 |
| D2 | LoRA 超参消融：r 取 8 / 16 / 64；目标模块比较 q,v 与全部线性层 | 在 CCDS 上运行中（9 个模型已完成 1 个） | 随时可做（CCDS） | 进行中 |
| D3 | 恢复意图识别和 Query 改写任务 | 在 CCDS 上运行中（第一次失败已修复，见 P20） | N1、N3 之后 | 进行中 |
| D4 | 蒸馏到 1.5B–3B 的小模型，用 vLLM 部署并压测 | 蒸馏与 transformers 测速已完成：推荐部署 3B gold（宏 F1 0.517，比教师低 3.1 个百分点），first 打分吞吐约 4 倍；vLLM 测速重跑中 | N1、N3 之后 | 进行中 |
| D5 | 中文扩展（Multi-CPR 电商子集） | 排队中，D2 结束后自动开始 | 英文主线全部完成后 | 进行中 |
