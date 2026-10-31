"""D4（DEVLOG S26）：蒸馏用的训练数据。教师是 E2a（seed 42）加先验调整，学生是 1.5B / 3B 的小模型。

教师标签 = argmax(log p + b)，b 是教师在调参集上学到的偏置（与 prior_adjust 相同的坐标上升），
也就是推荐方案在这条样本上的预测。三种学生训练集（验证集都用人工标注的 E2a 验证集，便于按同一标准选 checkpoint）：
- gold      ：E2a 的 2 万条人工标注（直接复用 e1_relevance，不在这里生成）；
- teacher20k：同样的 2 万条，标签换成教师标签；
- teacher50k：E4 的 5 万条嵌套训练集（包含上面的 2 万条，另外 3 万条来自其他 query），全部用教师标签。
  这 3 万条相当于“没有人工标注的新数据”，检验蒸馏能否靠教师标注的额外数据弥补小模型的差距。
输出：<out-dir>/{teacher20k,teacher50k}/{splits,llamafactory}/（prefix=esci_rel），<out-dir>/stats.json
（教师标签与人工标注的一致率、类别分布、教师偏置）。
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.data.to_alpaca import export
from ecomq.eval.prior_adjust import load, predict, tune_bias
from ecomq.schema import RELEVANCE


def relabel(recs: list[dict], L: np.ndarray, bias: np.ndarray) -> list[dict]:
    pred = predict(L, bias)
    out = []
    for r, p in zip(recs, pred):
        meta = {**r.get("meta", {}), "gold_relevance": r["label"]["relevance"], "label_source": "teacher-e2a-prior"}
        out.append({**r, "label": {"relevance": RELEVANCE[p]}, "meta": meta})
    return out


def summary(rows: list[dict]) -> dict:
    gold = [r["meta"]["gold_relevance"] for r in rows]
    teach = [r["label"]["relevance"] for r in rows]
    return {"rows": len(rows), "agree_with_gold": round(sum(g == t for g, t in zip(gold, teach)) / len(rows), 4),
            "teacher_label_share": {k: round(v / len(rows), 4) for k, v in Counter(teach).most_common()},
            "gold_label_share": {k: round(v / len(rows), 4) for k, v in Counter(gold).most_common()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="E2a 训练集（2 万条）")
    ap.add_argument("--pool", required=True, help="包含 base 的 5 万条嵌套训练集")
    ap.add_argument("--pool-scores", required=True, help="教师在 pool 上的标签打分（score_labels 输出）")
    ap.add_argument("--val", required=True, help="E2a 验证集（人工标注，原样使用）")
    ap.add_argument("--tune", required=True)
    ap.add_argument("--tune-scores", required=True, help="教师在调参集上的标签打分")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)

    _, Lt, gt = load(args.tune, args.tune_scores)
    bias, tune_f1 = tune_bias(Lt, gt)
    pool, Lp, _ = load(args.pool, args.pool_scores)
    labeled = relabel(pool, Lp, bias)
    base_ids = {r["id"] for r in read_jsonl(args.base)}
    t20 = [r for r in labeled if r["id"] in base_ids]
    assert len(t20) == len(base_ids), "pool must contain every base row"
    val = list(read_jsonl(args.val))
    stats = {"teacher_bias": dict(zip(RELEVANCE, bias.tolist())), "teacher_tune_macro_f1": round(tune_f1, 4)}
    for name, rows in (("teacher20k", t20), ("teacher50k", labeled)):
        write_jsonl(out / name / "splits" / "train.jsonl", rows)
        write_jsonl(out / name / "splits" / "val.jsonl", val)
        export(out / name / "splits", out / name / "llamafactory", prefix="esci_rel")
        stats[name] = summary(rows)
    extra = [r for r in labeled if r["id"] not in base_ids]
    stats["extra_rows_only"] = summary(extra)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
