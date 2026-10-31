"""D1–D5（DEVLOG S26）的纯 Python 部分：MMLU 提示与分批、BM25、重排指标、改写检索、蒸馏标签、标注解析、中文任务。"""
import json

import numpy as np

from ecomq.data.annotate_query import check, parse_annotation
from ecomq.data.build_d4 import relabel, summary
from ecomq.eval.mmlu import CATEGORY, build_prompt, paired_bootstrap, token_batches
from ecomq.eval.rerank import metrics, ranks_by_query
from ecomq.eval.rewrite_recall import evaluate
from ecomq.retrieval import BM25, char_ngrams, word_tokens
from ecomq.schema import LABELS, RELEVANCE_ZH, build_user_prompt, format_output, validate_record

Q = {"question": "2 + 2 = ?", "subject": "elementary_mathematics", "choices": ["3", "4", "5", "6"], "answer": 1}


def test_mmlu_prompt_ends_with_answer_and_shots_carry_answers():
    p = build_prompt(Q, [Q, Q])
    assert p.startswith("The following are multiple choice questions (with answers) about elementary mathematics.")
    assert p.count("Answer: B") == 2 and p.endswith("D. 6\nAnswer:")


def test_mmlu_has_57_subjects_in_four_categories():
    assert len(CATEGORY) == 57 and set(CATEGORY.values()) == {"STEM", "humanities", "social_sciences", "other"}


def test_token_batches_respect_budget_and_cover_all():
    lengths = [10, 500, 30, 30, 800, 20]
    batches = token_batches(lengths, max_tokens=1000, max_batch=3)
    assert sorted(i for b in batches for i in b) == list(range(6))
    for b in batches:
        assert len(b) <= 3 and (len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 1000)


def test_paired_bootstrap_sign():
    r = paired_bootstrap([True] * 50 + [False] * 50, [True] * 60 + [False] * 40, n_boot=200)
    assert r["diff"] == 0.1 and r["ci95"][0] <= 0.1 <= r["ci95"][1]


def test_bm25_ranks_matching_document_first():
    docs = ["红色 连衣裙 夏季", "黑色 运动鞋 男款", "红色 运动鞋 女款"]
    top = BM25(docs, char_ngrams).topk(["红色运动鞋"], 3)[0]
    assert top[0][0] == 2 and len(top) == 3
    en = BM25(["nike running shoes", "iphone case", "running socks"], word_tokens).topk(["running shoes"], 2)[0]
    assert en[0][0] == 0


def _dev():
    rows = []
    for q, pos in (("a", 3), ("b", None)):
        for r in range(1, 5):
            rows.append({"id": f"{q}{r}", "group": q, "meta": {"bm25_rank": r},
                         "label": {"relevance": "relevant" if r == pos else "irrelevant"}})
    return rows


def test_rerank_ranks_and_metrics():
    dev = _dev()
    assert ranks_by_query(dev, None) == {"a": 3, "b": None}
    score = {r["id"]: (5.0 if r["id"] == "a3" else 0.0) for r in dev}
    ranks = ranks_by_query(dev, score)
    assert ranks == {"a": 1, "b": None}
    assert metrics(list(ranks.values()))["mrr@10"] == 0.5


def test_rewrite_recall_evaluate():
    rec, rr = evaluate([[1, 2, 3], [4, 5, 6]], [{2, 9}, {7}], k=2)
    assert rec == [0.5, 0.0] and rr == [0.5, 0.0]


def test_relabel_uses_bias_and_keeps_gold():
    recs = [{"id": "x", "label": {"relevance": "exact"}, "meta": {}},
            {"id": "y", "label": {"relevance": "complement"}, "meta": {}}]
    L = np.array([[-0.1, -3.0, -2.0, -5.0], [-0.5, -3.0, -1.0, -5.0]])
    out = relabel(recs, L, np.array([0.0, 0.0, 0.0, 0.0]))
    assert [r["label"]["relevance"] for r in out] == ["exact", "exact"]
    out = relabel(recs, L, np.array([0.0, 0.0, 1.0, 0.0]))
    assert out[1]["label"]["relevance"] == "complement" and out[1]["meta"]["gold_relevance"] == "complement"
    assert summary(out)["agree_with_gold"] == 1.0


def test_parse_and_check_annotation():
    q = {"query_id": 1, "query": "nike mens running shoes size 10"}
    good = json.dumps({"intent": "product_search", "attributes": {"brand": "nike", "size": "10"},
                       "rewrite": "nike men's running shoes size 10"})
    label, errors = check(q, good)
    assert label and label["relevance"] is None and not errors
    bad = json.dumps({"intent": "product_search", "attributes": {"brand": "adidas"}, "rewrite": "x"})
    label, errors = check(q, bad)
    assert label is None and any("adidas" in e for e in errors)
    assert parse_annotation("not json") is None


def test_relevance_zh_task():
    rec = {"id": "m", "task": "relevance_zh", "query": "红色运动鞋", "title": "女款红色跑步鞋",
           "label": {"relevance": "relevant"}}
    assert validate_record(rec) == []
    assert validate_record({**rec, "label": {"relevance": "exact"}})
    assert LABELS["relevance_zh"] == RELEVANCE_ZH
    assert format_output({"relevance": "irrelevant"}, "relevance_zh") == '{"relevance": "irrelevant"}'
    assert "电商搜索" in build_user_prompt("红色运动鞋", "女款红色跑步鞋", "relevance_zh")
