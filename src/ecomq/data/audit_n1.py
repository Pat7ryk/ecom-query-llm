"""Audit the four immutable datasets used by N0/N1 and write a hash manifest.

This module never creates a new split.  It only verifies the outputs produced by
``prepare_esci`` so a run cannot silently proceed with query leakage or changed
input files.
"""
import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from ecomq.data.io import read_jsonl
from ecomq.paths import redact


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_stats(path: Path) -> tuple[dict, set[int | str]]:
    rows = list(read_jsonl(path))
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), f"duplicate ids in {path}"
    query_ids = {
        (row.get("meta") or {}).get("query_id", row.get("group", row["id"]))
        for row in rows
    }
    labels = Counter(row["label"]["relevance"] for row in rows)
    return {
        "path": redact(str(path)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "queries": len(query_ids),
        "labels": dict(sorted(labels.items())),
    }, query_ids


def audit_datasets(paths: dict[str, Path]) -> dict:
    expected = {"train", "val", "tune", "test"}
    assert set(paths) == expected, f"expected {sorted(expected)}, got {sorted(paths)}"
    stats, queries = {}, {}
    for name, path in paths.items():
        assert path.is_file(), f"missing dataset: {path}"
        stats[name], queries[name] = _split_stats(path)

    overlaps = {}
    for left, right in combinations(sorted(paths), 2):
        shared = queries[left] & queries[right]
        if shared:
            overlaps[f"{left}__{right}"] = {
                "count": len(shared),
                "examples": sorted(map(str, shared))[:10],
            }
    assert not overlaps, f"query leakage detected: {overlaps}"
    return {"splits": stats, "query_overlap": overlaps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.data_root)
    paths = {
        "train": root / "e1_relevance/splits/train.jsonl",
        "val": root / "e1_relevance/splits/val.jsonl",
        "tune": root / "esci_tune/splits/tune.jsonl",
        "test": root / "esci_test2500/splits/test.jsonl",
    }
    report = audit_datasets(paths)
    expected = {
        "train": {"rows": 20000, "queries": 5000},
        "val": {"rows": 1000, "queries": 250},
        "tune": {"rows": 6000, "queries": 1500},
        "test": {"rows": 10000, "queries": 2500},
    }
    observed = {
        name: {key: report["splits"][name][key] for key in ("rows", "queries")}
        for name in expected
    }
    assert observed == expected, f"N1 dataset sizes changed: {observed}"
    report["expected_sizes"] = expected
    raw = root / "raw/esci"
    report["raw_files"] = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(raw.glob("*")) if path.is_file()
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
