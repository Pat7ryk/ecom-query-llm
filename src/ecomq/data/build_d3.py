"""D3（DEVLOG S26）：多任务与单任务的训练数据。

- mt（多任务）：E2a 的相关性训练集（2 万条）+ 这些训练 query 的 full 标注（每个 query 一条）；验证集同样合并。
  与 E2a 只差“多了一个 query 理解任务”，用来回答提案里的问题：统一建模会不会伤害（或帮助）相关性判断。
- full_only（单任务）：只有 full 标注，用来与 mt 比较 full 任务本身的质量。
full 标注来自 annotate_query，按 meta.query_id 与相关性数据对齐；标注时被丢弃的 query 不进入训练集。
输出：<out-dir>/{mt,full_only}/{splits,llamafactory}/（prefix=esci_rel），<out-dir>/stats.json。
"""
import argparse
import json
import random
from pathlib import Path

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.data.to_alpaca import export


def split_full(full: list[dict], rel: list[dict]) -> list[dict]:
    qids = {r["meta"]["query_id"] for r in rel}
    return [f for f in full if f["meta"]["query_id"] in qids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel-dir", required=True, help="E2a 的 splits 目录（train.jsonl、val.jsonl）")
    ap.add_argument("--full", required=True, help="annotate_query 对训练与验证 query 的标注")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rel_dir, out = Path(args.rel_dir), Path(args.out_dir)
    full = list(read_jsonl(args.full))
    stats = {}
    for name in ("train", "val"):
        rel = list(read_jsonl(rel_dir / f"{name}.jsonl"))
        f = split_full(full, rel)
        mixed = rel + f
        random.Random(args.seed).shuffle(mixed)
        write_jsonl(out / "mt" / "splits" / f"{name}.jsonl", mixed)
        write_jsonl(out / "full_only" / "splits" / f"{name}.jsonl", f)
        stats[name] = {"relevance_rows": len(rel), "full_rows": len(f),
                       "queries": len({r["meta"]["query_id"] for r in rel})}
    for v in ("mt", "full_only"):
        export(out / v / "splits", out / v / "llamafactory", prefix="esci_rel")
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
