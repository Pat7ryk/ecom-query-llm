# D3 标注抽查（DEVLOG S26 结果（二））

从 `annot_test.jsonl`（7B 基座零样本标注，2,435 条通过规则校验）中用 `random.seed(26)` 抽 100 条，记录 id 见 `ids.json`。
由 Claude 逐条判断意图、属性、改写是否合理。这不是人工标注，只用来粗略估计标注质量。

| 项目 | 结果 | 典型问题 |
|---|---|---|
| 意图 | 94 / 100 合理 | 6 条错误全是把具体商品判成 other：游戏名（the division 2）、书名（god loves you little one）、专辑（jonghyun poet artist album）、零件号（um09e71）、拼写错误的商品（wirless ierphobes）、剧名（blacklist） |
| 属性：值错误 | 12 / 100 至少有一个值放错了字段 | brand="battery"、gender="cats"（battery cat toys）；color="colors"；model="unlocked"；model="steel toe boots"（应为 category）；category="sexy"；把作者或专辑名当成品牌或品类 |
| 属性：明显遗漏 | 约 17 / 100 漏掉了 query 里写明的品牌或尺寸 | google、kef、klairs、alli、footjoy 等品牌；12 x 18、6 gallon、55” 等尺寸；category 时有时无，标准不一致 |
| 改写 | 2 条丢掉了约束，约 6 条漏改了拼写错误 | “coffee mug with g” 改成了 “coffee mug”；“samsung galaxy j2 case never give up” 丢掉了图案词；beenie、biys、toddle、sherrif、cotapaxi 没有改正 |

结论：意图标签 91% 是 product_search，所以意图准确率高并不说明标得好；属性的召回明显不足，值也有约 12% 的错误。
用这份标注训练出的模型，学到的上限就是这个水平（见 N4：在 NSCC 上换更强的标注模型重做）。
