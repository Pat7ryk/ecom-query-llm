"""D3（DEVLOG S26）：为真实的 ESCI query 标注意图、属性和改写（full 任务），作为多任务训练与评测的数据。

原计划用强模型从商品标题反向合成 query（ecomq.data.synth，需要外部 API）。这里改为直接标注 ESCI 里真实的 query：
query 的分布与相关性任务一致，也不需要 API。标注模型是本地的开源模型（默认 Qwen2.5-14B-Instruct，比被训练的 7B 大一档），
贪心解码，输出 {"intent", "attributes", "rewrite"}；没有商品标题，所以 relevance 为 null。
每条输出用 schema.validate_record 做规则校验（意图在标签集内、属性值必须原样出现在 query 里、改写非空）；
不合格的把错误原因附在提示后重试一次，仍不合格的丢弃并计数。
输入：relevance 记录（一个或多个 jsonl），按 meta.query_id 去重后每个 query 标注一次。
输出：<output>（full 任务记录）与 <output>.report.json（通过率、重试、意图与属性分布、耗时）。
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.schema import ATTR_KEYS, INTENTS, validate_record

PROMPT = f"""You annotate English e-commerce search queries for a query understanding model.
For the query below, output a JSON object with exactly these fields:
- "intent": one of {", ".join(INTENTS)}. product_search = looking for a product to buy; compare = comparing products or prices;
  after_sales = returns, repairs, orders; inquiry = a question about a product or topic; other = anything else.
- "attributes": an object whose keys come only from [{", ".join(ATTR_KEYS)}]. Each value must be copied exactly as it
  appears in the query (same spelling). Include only attributes the query actually states; use {{{{}}}} if there are none.
  category is the kind of product (for example "running shoes"); price_min / price_max only for explicit prices.
- "rewrite": the query rewritten as a clean search string: fix spelling mistakes, expand abbreviations, drop filler words,
  keep every constraint (brand, size, color, quantity, negation). If the query is already clean, repeat it unchanged.
Output the JSON object only.

Query: {{query}}"""

RETRY = "\n\nYour previous answer was rejected: {errors}. Fix these problems and output the JSON object again."


def unique_queries(paths: list[str]) -> list[dict]:
    seen, out = set(), []
    for path in paths:
        for r in read_jsonl(path):
            qid = r["meta"]["query_id"]
            if qid not in seen:
                seen.add(qid)
                out.append({"query_id": qid, "query": r["query"]})
    return out


def parse_annotation(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not {"intent", "attributes", "rewrite"} <= obj.keys():
        return None
    return {"intent": obj["intent"], "attributes": obj["attributes"], "rewrite": obj["rewrite"], "relevance": None}


def to_record(q: dict, label: dict, teacher: str, retried: bool) -> dict:
    return {"id": f"d3-{q['query_id']}", "task": "full", "query": q["query"], "label": label,
            "meta": {"query_id": q["query_id"], "source": "esci-query", "teacher": teacher, "retried": retried}}


def check(q: dict, text: str) -> tuple[dict | None, list[str]]:
    label = parse_annotation(text)
    if label is None:
        return None, ["output is not a JSON object with intent, attributes, rewrite"]
    rec = to_record(q, label, "", False)
    errors = validate_record(rec)
    return (label, errors) if not errors else (None, errors)


def report(records: list[dict], dropped: int, n: int, seconds: float) -> dict:
    labels = [r["label"] for r in records]
    attr_keys = Counter(k for lab in labels for k in lab["attributes"])
    return {
        "queries": n, "kept": len(records), "dropped": dropped,
        "retried_and_kept": sum(r["meta"]["retried"] for r in records),
        "intent": dict(Counter(lab["intent"] for lab in labels).most_common()),
        "attribute_keys": dict(attr_keys.most_common()),
        "queries_with_attributes": round(sum(bool(lab["attributes"]) for lab in labels) / max(len(labels), 1), 4),
        "rewrite_changed": round(sum(lab["rewrite"].strip().lower() != r["query"].strip().lower()
                                     for lab, r in zip(labels, records)) / max(len(labels), 1), 4),
        "seconds": round(seconds), "queries_per_second": round(n / max(seconds, 1e-9), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", nargs="+", required=True, help="relevance 记录 jsonl（按 query 去重）")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    queries = unique_queries(args.inputs)
    if args.limit:
        queries = queries[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    def generate(contents: list[str]) -> list[str]:
        out = []
        for s in range(0, len(contents), args.batch_size):
            prompts = [tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                               add_generation_prompt=True) for c in contents[s : s + args.batch_size]]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
            out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out

    start = time.time()
    first = [PROMPT.format(query=q["query"]) for q in queries]
    texts = generate(first)
    results, retry_idx, retry_prompts = [None] * len(queries), [], []
    for i, (q, t) in enumerate(zip(queries, texts)):
        label, errors = check(q, t)
        if label:
            results[i] = to_record(q, label, Path(args.model).name, False)
        else:
            retry_idx.append(i)
            retry_prompts.append(first[i] + RETRY.format(errors="; ".join(errors)))
    for i, t in zip(retry_idx, generate(retry_prompts)):
        label, _ = check(queries[i], t)
        if label:
            results[i] = to_record(queries[i], label, Path(args.model).name, True)
    records = [r for r in results if r]
    write_jsonl(args.output, records)
    rep = report(records, len(queries) - len(records), len(queries), time.time() - start)
    Path(args.output + ".report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
