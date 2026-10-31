"""D5（DEVLOG S26）：中文扩展。Multi-CPR 电商子集 → relevance_zh 二分类训练数据 + BM25 重排评测集。

Multi-CPR（阿里巴巴，SIGIR 2022）电商子集：约 100 万商品标题（corpus.tsv）、10 万条训练 query、1,000 条开发 query；
每个 query 只标注了一个相关商品（qrels），没有标注不相关商品。所以：
- 检索用 BM25，分词用字的一元和二元组合（中文商品标题很短，字 n-gram 不需要分词词典，也没有额外依赖）。
- 训练集：抽 --train-queries 个训练 query，每个 query 取 1 个正例（qrels）和 1 个难负例：
  从 BM25 第 --neg-from 到 --neg-to 名中随机取一个非正例商品。只标了一个正例，难负例里可能混有其实相关的商品
  （假负例）；跳过最靠前的几名可以减少这种情况，比例写进统计，评测时也要记住这一点。
- 开发集（评测用）：1,000 个开发 query 各取 BM25 前 --rerank-k 名作为候选，标注正例为 relevant、其余为 irrelevant，
  评测时按 query 重排、计算 MRR@10（ecomq.eval.rerank）。正例不在候选里的 query 照样保留（重排的上限 = BM25 的召回率）。
输出：<out-dir>/splits/{train,val,dev_rerank}.jsonl、<out-dir>/llamafactory/（prefix=esci_rel，可直接用 E2a 的配置）、
     <out-dir>/stats.json。
"""
import argparse
import json
import random
from pathlib import Path

from ecomq.data.io import write_jsonl
from ecomq.data.to_alpaca import export
from ecomq.retrieval import BM25, char_ngrams
from ecomq.schema import validate_record


def read_tsv(path: Path, n_cols: int) -> list[list[str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= n_cols:
                rows.append(parts)
    return rows


def record(qid: str, query: str, pid: str, title: str, label: str, split: str, **meta) -> dict:
    return {"id": f"mcpr-{split}-{qid}-{pid}", "task": "relevance_zh", "query": query, "title": title,
            "label": {"relevance": label}, "group": f"mcpr-{qid}", "meta": {"source": "multicpr-ecom", **meta}}


def mrr_at(ranks: list[int | None], k: int = 10) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="Multi-CPR data/ecom 目录")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-queries", type=int, default=10000)
    ap.add_argument("--val-queries", type=int, default=250)
    ap.add_argument("--neg-from", type=int, default=5, help="难负例从 BM25 第几名开始取（1 起计）")
    ap.add_argument("--neg-to", type=int, default=30)
    ap.add_argument("--rerank-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    rng = random.Random(args.seed)

    corpus = read_tsv(raw / "corpus.tsv", 2)
    pids = [r[0] for r in corpus]
    titles = [r[1] for r in corpus]
    pos_of = {pid: i for i, pid in enumerate(pids)}
    train_q = dict(read_tsv(raw / "train.query.txt", 2))
    dev_q = dict(read_tsv(raw / "dev.query.txt", 2))
    qrels_train = {r[0]: r[2] for r in read_tsv(raw / "qrels.train.tsv", 4)}
    qrels_dev = {r[0]: r[2] for r in read_tsv(raw / "qrels.dev.tsv", 4)}

    qids = sorted(q for q in train_q if q in qrels_train)
    rng.shuffle(qids)
    picked = qids[: args.train_queries + args.val_queries]
    dev_ids = sorted(dev_q, key=int)

    bm25 = BM25(titles, char_ngrams)
    hits = bm25.topk([train_q[q] for q in picked] + [dev_q[q] for q in dev_ids], max(args.neg_to, args.rerank_k))
    train_hits, dev_hits = hits[: len(picked)], hits[len(picked):]

    splits = {"train": [], "val": []}
    pos_rank_train = []
    for n, (qid, hit) in enumerate(zip(picked, train_hits)):
        split = "train" if n < args.train_queries else "val"
        pos = qrels_train[qid]
        ranked = [pids[j] for j, _ in hit]
        pos_rank_train.append(ranked.index(pos) + 1 if pos in ranked else None)
        pool = [p for p in ranked[args.neg_from - 1 : args.neg_to] if p != pos]
        neg = rng.choice(pool)
        q = train_q[qid]
        splits[split].append(record(qid, q, pos, titles[pos_of[pos]], "relevant", split))
        splits[split].append(record(qid, q, neg, titles[pos_of[neg]], "irrelevant", split,
                                    bm25_rank=ranked.index(neg) + 1))
    for rows in splits.values():
        rng.shuffle(rows)

    dev_rows, dev_ranks = [], []
    for qid, hit in zip(dev_ids, dev_hits):
        pos = qrels_dev[qid]
        cands = hit[: args.rerank_k]
        ranked = [pids[j] for j, _ in cands]
        dev_ranks.append(ranked.index(pos) + 1 if pos in ranked else None)
        for r, (j, s) in enumerate(cands, 1):
            dev_rows.append(record(qid, dev_q[qid], pids[j], titles[j], "relevant" if pids[j] == pos else "irrelevant",
                                   "dev", bm25_rank=r, bm25_score=round(s, 4), positive_in_candidates=pos in ranked))

    for rows in (*splits.values(), dev_rows):
        bad = [r["id"] for r in rows if validate_record(r)]
        assert not bad, bad[:5]
    for name, rows in (*splits.items(), ("dev_rerank", dev_rows)):
        write_jsonl(out / "splits" / f"{name}.jsonl", rows)
    export(out / "splits", out / "llamafactory", prefix="esci_rel")

    in_top = lambda ranks, k: round(sum(r is not None and r <= k for r in ranks) / len(ranks), 4)  # noqa: E731
    stats = {
        "corpus": len(pids), "train_queries": args.train_queries, "val_queries": args.val_queries,
        "train_rows": len(splits["train"]), "val_rows": len(splits["val"]), "dev_queries": len(dev_ids),
        "dev_rows": len(dev_rows), "neg_ranks": [args.neg_from, args.neg_to], "rerank_k": args.rerank_k,
        "bm25_dev": {"mrr@10": round(mrr_at(dev_ranks), 4), "recall@1": in_top(dev_ranks, 1),
                     "recall@10": in_top(dev_ranks, 10), f"recall@{args.rerank_k}": in_top(dev_ranks, args.rerank_k)},
        "bm25_train_positive_recall@30": in_top(pos_rank_train, 30),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
