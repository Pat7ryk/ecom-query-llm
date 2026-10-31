"""D3（DEVLOG S26）：改写对检索的帮助。用 BM25 在测试集的全部商品标题（约 1 万个）上检索，
看每个 query 标注为 exact 的商品能否排进前 k 名，比较“原始 query”与各系统的改写。

- 语料：2500 query 测试集中出现的所有商品（按 product_id 去重）。
- 相关商品：该 query 下标注为 exact 的商品；没有 exact 商品的 query 不参与。
- 指标：Recall@10（找回的 exact 商品占比，按 query 平均）、MRR@10（第一个 exact 商品的名次）。
  差值用按 query 的 bootstrap 给出 95% 置信区间。
局限：测试集的商品本来就是 Amazon 搜索针对原始 query 召回的，原始 query 天然占优，所以这里只看相对变化。
系统的改写来自 generate_hf 的预测（id 为 d3-<query_id>，输出为 full 任务的 JSON）或 annotate_query 的标注。
  python -m ecomq.eval.rewrite_recall --test test.jsonl --systems teacher=ann.jsonl mt=preds.jsonl --output r.json
"""
import argparse
import json
import random
from collections import defaultdict

from ecomq.data.io import read_jsonl
from ecomq.retrieval import BM25, word_tokens
from ecomq.schema import parse_output


def load_rewrites(path: str) -> dict:
    """返回 {query_id: 改写}；解析失败的条目不返回（评测时退回原始 query，并计入 fallback 数）。"""
    out = {}
    for r in read_jsonl(path):
        if "label" in r:  # annotate_query 的标注
            out[r["meta"]["query_id"]] = r["label"]["rewrite"]
            continue
        parsed = parse_output(r["output"], "full")
        if parsed and isinstance(parsed.get("rewrite"), str) and parsed["rewrite"].strip():
            out[int(r["id"].split("-", 1)[1])] = parsed["rewrite"]
    return out


def evaluate(ranked: list[list[int]], relevant: list[set[int]], k: int = 10) -> tuple[list[float], list[float]]:
    rec, rr = [], []
    for top, rel in zip(ranked, relevant):
        top = top[:k]
        rec.append(sum(p in rel for p in top) / len(rel))
        first = next((i for i, p in enumerate(top, 1) if p in rel), None)
        rr.append(1 / first if first else 0.0)
    return rec, rr


def bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(d)
    stats = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"diff": round(sum(d) / n, 4), "ci95": [round(stats[int(0.025 * n_boot)], 4),
                                                   round(stats[int(0.975 * n_boot) - 1], 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="2500 query 测试集（relevance 记录）")
    ap.add_argument("--systems", nargs="+", required=True, help="name=path（改写来源）")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    test = list(read_jsonl(args.test))
    pids, titles, index = [], [], {}
    for r in test:
        p = r["meta"]["product_id"]
        if p not in index:
            index[p] = len(pids)
            pids.append(p)
            titles.append(r["title"])
    queries, exact = {}, defaultdict(set)
    for r in test:
        queries[r["meta"]["query_id"]] = r["query"]
        if r["label"]["relevance"] == "exact":
            exact[r["meta"]["query_id"]].add(index[r["meta"]["product_id"]])
    qids = sorted(q for q in queries if exact[q])
    bm25 = BM25(titles, word_tokens)

    def run(texts):
        hits = bm25.topk(texts, args.k)
        return evaluate([[j for j, _ in h] for h in hits], [exact[q] for q in qids], args.k)

    base_rec, base_rr = run([queries[q] for q in qids])
    res = {"queries_with_exact": len(qids), "corpus": len(pids), "k": args.k,
           "original": {"recall": round(sum(base_rec) / len(qids), 4), "mrr": round(sum(base_rr) / len(qids), 4)}}
    for spec in args.systems:
        name, path = spec.split("=", 1)
        rw = load_rewrites(path)
        texts = [rw.get(q, queries[q]) for q in qids]
        rec, rr = run(texts)
        res[name] = {"recall": round(sum(rec) / len(qids), 4), "mrr": round(sum(rr) / len(qids), 4),
                     "fallback_to_original": sum(q not in rw for q in qids),
                     "changed": sum(rw.get(q, queries[q]).strip().lower() != queries[q].strip().lower() for q in qids),
                     "recall_diff": bootstrap(base_rec, rec), "mrr_diff": bootstrap(base_rr, rr)}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
