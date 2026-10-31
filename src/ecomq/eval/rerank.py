"""D5（DEVLOG S26）：Multi-CPR 开发集的重排评测。

每个 query 有 BM25 前 k 名候选（prepare_multicpr 的 dev_rerank.jsonl），其中最多一个是标注的相关商品。
重排分数 = log p(relevant) − log p(irrelevant)（score_labels --task relevance_zh 的输出）。这个差值对整体偏置不变，
所以只比较排序能力，不受模型偏向哪一类的影响。指标：MRR@10、Recall@1/5/10（都在 k 个候选内）；
差值用按 query 的 bootstrap 给出 95% 置信区间。另报二分类的 AUC（全部候选对）作参考。
  python -m ecomq.eval.rerank --dev dev_rerank.jsonl --systems bm25 zeroshot=a.jsonl d5=b.jsonl --output report.json
"""
import argparse
import json
import random
from collections import defaultdict

from ecomq.data.io import read_jsonl


def ranks_by_query(dev: list[dict], score: dict[str, float] | None) -> dict[str, int | None]:
    """每个 query 中正例的名次（1 起计）；正例不在候选里为 None。score 为 None 时用 BM25 原始名次。"""
    groups = defaultdict(list)
    for r in dev:
        groups[r["group"]].append(r)
    out = {}
    for g, rows in groups.items():
        if score is None:
            order = sorted(rows, key=lambda r: r["meta"]["bm25_rank"])
        else:  # 分数相同时按 BM25 名次，结果可复现
            order = sorted(rows, key=lambda r: (-score[r["id"]], r["meta"]["bm25_rank"]))
        pos = [i for i, r in enumerate(order, 1) if r["label"]["relevance"] == "relevant"]
        out[g] = pos[0] if pos else None
    return out


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    hit = lambda k: sum(r is not None and r <= k for r in ranks) / n  # noqa: E731
    return {"mrr@10": round(sum(1 / r for r in ranks if r is not None and r <= 10) / n, 4),
            "recall@1": round(hit(1), 4), "recall@5": round(hit(5), 4), "recall@10": round(hit(10), 4)}


def rr(r):
    return 1 / r if r is not None and r <= 10 else 0.0


def bootstrap_diff(a: dict, b: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    keys = sorted(a)
    d = [rr(b[k]) - rr(a[k]) for k in keys]
    rng = random.Random(seed)
    n = len(d)
    stats = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mrr@10_diff": round(sum(d) / n, 4),
            "ci95": [round(stats[int(0.025 * n_boot)], 4), round(stats[int(0.975 * n_boot) - 1], 4)]}


def auc(dev: list[dict], score: dict[str, float]) -> float:
    pos = [score[r["id"]] for r in dev if r["label"]["relevance"] == "relevant"]
    neg = sorted(score[r["id"]] for r in dev if r["label"]["relevance"] != "relevant")
    import bisect

    wins = sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p))
               for p in pos)
    return round(wins / (len(pos) * len(neg)), 4)


def load_scores(path: str) -> dict[str, float]:
    return {r["id"]: r["logp"]["relevant"] - r["logp"]["irrelevant"] for r in read_jsonl(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--systems", nargs="+", required=True, help="bm25 或 name=scores.jsonl（同名多个种子用逗号分隔路径）")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    dev = list(read_jsonl(args.dev))
    ranks, res = {}, {"n_queries": len({r["group"] for r in dev}), "systems": {}}
    for spec in args.systems:
        name, _, paths = spec.partition("=")
        if not paths:
            ranks[name] = ranks_by_query(dev, None)
            res["systems"][name] = metrics(list(ranks[name].values()))
            continue
        per_seed = []
        for i, p in enumerate(paths.split(",")):
            s = load_scores(p)
            key = name if len(paths.split(",")) == 1 else f"{name}-{i}"
            ranks[key] = ranks_by_query(dev, s)
            per_seed.append({**metrics(list(ranks[key].values())), "auc": auc(dev, s)})
            res["systems"][key] = per_seed[-1]
        if len(per_seed) > 1:
            res["systems"][name] = {k: {"mean": round(sum(m[k] for m in per_seed) / len(per_seed), 4),
                                        "range": [min(m[k] for m in per_seed), max(m[k] for m in per_seed)]}
                                    for k in per_seed[0]}
    base = next(iter(ranks))
    res["diff_vs_" + base] = {k: bootstrap_diff(ranks[base], v) for k, v in ranks.items() if k != base}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res["systems"]))


if __name__ == "__main__":
    main()
