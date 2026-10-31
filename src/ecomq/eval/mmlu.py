"""D1（DEVLOG S26）：通用能力回归抽测。MMLU 5-shot，比较基座与 E2a 适配器，回答“微调是否明显削弱了通用能力”。

做法与 MMLU 原论文和 lm-eval 的常见设定一致：
- 提示：“The following are multiple choice questions (with answers) about <学科>.” + 该学科 dev 集的 5 道例题
  + 待测题目，以 “Answer:” 结尾；不套对话模板，两个模型用完全相同的文本。
- 判分：只看下一个 token 在 “ A” “ B” “ C” “ D” 四个 token 上的 logits，取最大者，不让模型自由生成。
- 与 score_labels 一样使用右侧填充，取每条序列最后一个真实 token 位置的 logits。
数据：Hugging Face 上 cais/mmlu 的 all/{dev,test} parquet（question, subject, choices, answer）。
  python -m ecomq.eval.mmlu run --model <dir> [--lora <dir>] --load-4bit --data-dir <mmlu>/all --output <x.jsonl>
  python -m ecomq.eval.mmlu compare --a base.jsonl --b e2a.jsonl --output report.json
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from ecomq.data.io import read_jsonl, write_jsonl

LETTERS = ["A", "B", "C", "D"]

# MMLU 原论文（Hendrycks 等）的四大类；学科到大类的对应按其 categories.py 的子类归并
CATEGORY = {}
for _cat, _subjects in {
    "STEM": "abstract_algebra astronomy college_biology college_chemistry college_computer_science college_mathematics "
            "college_physics computer_security conceptual_physics electrical_engineering elementary_mathematics "
            "high_school_biology high_school_chemistry high_school_computer_science high_school_mathematics "
            "high_school_physics high_school_statistics machine_learning",
    "humanities": "formal_logic high_school_european_history high_school_us_history high_school_world_history "
                  "international_law jurisprudence logical_fallacies moral_disputes moral_scenarios philosophy "
                  "prehistory professional_law world_religions",
    "social_sciences": "econometrics high_school_geography high_school_government_and_politics "
                       "high_school_macroeconomics high_school_microeconomics high_school_psychology human_sexuality "
                       "professional_psychology public_relations security_studies sociology us_foreign_policy",
    "other": "anatomy business_ethics clinical_knowledge college_medicine global_facts human_aging management "
             "marketing medical_genetics miscellaneous nutrition professional_accounting professional_medicine virology",
}.items():
    for _s in _subjects.split():
        CATEGORY[_s] = _cat


def format_question(q: dict, with_answer: bool) -> str:
    text = q["question"].strip()
    for letter, choice in zip(LETTERS, q["choices"]):
        text += f"\n{letter}. {choice}"
    text += "\nAnswer:"
    if with_answer:
        text += f" {LETTERS[int(q['answer'])]}\n\n"
    return text


def build_prompt(q: dict, shots: list[dict]) -> str:
    subject = q["subject"].replace("_", " ")
    head = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    return head + "".join(format_question(s, True) for s in shots) + format_question(q, False)


def load_split(data_dir: str, split: str) -> list[dict]:
    import pandas as pd

    files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    assert files, f"no {split} parquet under {data_dir}"
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return [{"question": r.question, "subject": r.subject, "choices": list(r.choices), "answer": int(r.answer)}
            for r in df.itertuples()]


def token_batches(lengths: list[int], max_tokens: int, max_batch: int) -> list[list[int]]:
    """按长度排序后分批，每批的 (最长序列 × 条数) 不超过 max_tokens，减少填充、避免长题目显存溢出。"""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[i] > max_tokens or len(cur) >= max_batch:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    test, dev = load_split(args.data_dir, "test"), load_split(args.data_dir, "dev")
    shots = defaultdict(list)
    for q in dev:
        shots[q["subject"]].append(q)
    if args.limit:
        random.Random(0).shuffle(test)
        test = test[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    letter_ids = []
    for letter in LETTERS:
        ids = tok(" " + letter, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"' {letter}' is not a single token: {ids}"
        letter_ids.append(ids[0])
    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    seqs = [tok(build_prompt(q, shots[q["subject"]][: args.shots]), add_special_tokens=False)["input_ids"]
            for q in test]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = [None] * len(test)
    for batch in token_batches([len(s) for s in seqs], args.max_tokens, args.max_batch):
        width = max(len(seqs[i]) for i in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for j, i in enumerate(batch):
            ids[j, : len(seqs[i])] = torch.tensor(seqs[i])
            att[j, : len(seqs[i])] = 1
        with torch.no_grad():
            logits = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).logits
        for j, i in enumerate(batch):
            lg = logits[j, len(seqs[i]) - 1, letter_ids].float()
            pred = int(lg.argmax())
            out[i] = {"subject": test[i]["subject"], "answer": test[i]["answer"], "pred": pred,
                      "correct": pred == test[i]["answer"], "logits": [round(x, 4) for x in lg.tolist()]}
    write_jsonl(args.output, [{"qid": i, **o} for i, o in enumerate(out)])
    summary = summarize(out)
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "accuracy", "by_category")}))


def summarize(rows: list[dict]) -> dict:
    by_subject, by_cat = defaultdict(list), defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r["correct"])
        by_cat[CATEGORY.get(r["subject"], "other")].append(r["correct"])
    acc = lambda xs: round(sum(xs) / len(xs), 4)  # noqa: E731
    return {"n": len(rows), "accuracy": acc([r["correct"] for r in rows]),
            "macro_subject_accuracy": round(sum(acc(v) for v in by_subject.values()) / len(by_subject), 4),
            "by_category": {k: acc(v) for k, v in sorted(by_cat.items())},
            "by_subject": {k: acc(v) for k, v in sorted(by_subject.items())}}


def paired_bootstrap(a: list[bool], b: list[bool], n_boot: int = 2000, seed: int = 0) -> dict:
    """b − a 的准确率差及 95% 置信区间（对题目有放回重采样；两边是同一批题目）。"""
    assert len(a) == len(b)
    rng = random.Random(seed)
    n = len(a)
    d = [int(y) - int(x) for x, y in zip(a, b)]
    stats = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"diff": round(sum(d) / n, 4), "ci95": [round(stats[int(0.025 * n_boot)], 4),
                                                   round(stats[int(0.975 * n_boot) - 1], 4)]}


def compare(args):
    a = {r["qid"]: r for r in read_jsonl(args.a)}
    res = {"a": args.a_name, "summary_a": summarize(list(a.values())), "b": {}}
    for name, path in (x.split("=", 1) for x in args.b):
        b = {r["qid"]: r for r in read_jsonl(path)}
        assert a.keys() == b.keys() and all(a[k]["answer"] == b[k]["answer"] for k in a), "different question sets"
        keys = sorted(a)
        per_cat = {}
        for cat in sorted(set(CATEGORY.values())):
            ks = [k for k in keys if CATEGORY.get(a[k]["subject"], "other") == cat]
            per_cat[cat] = paired_bootstrap([a[k]["correct"] for k in ks], [b[k]["correct"] for k in ks],
                                            n_boot=1000)
        res["b"][name] = {"summary": {k: v for k, v in summarize(list(b.values())).items() if k != "by_subject"},
                          "diff_vs_a": paired_bootstrap([a[k]["correct"] for k in keys], [b[k]["correct"] for k in keys]),
                          "diff_by_category": per_cat,
                          "agreement_with_a": round(sum(a[k]["pred"] == b[k]["pred"] for k in keys) / len(keys), 4)}
    Path(args.output).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({n: (v["summary"]["accuracy"], v["diff_vs_a"]) for n, v in res["b"].items()}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--lora")
    r.add_argument("--load-4bit", action="store_true")
    r.add_argument("--data-dir", required=True, help="cais/mmlu 的 all/ 目录（dev 与 test 的 parquet）")
    r.add_argument("--output", required=True)
    r.add_argument("--shots", type=int, default=5)
    r.add_argument("--max-tokens", type=int, default=24000, help="每批 token 上限（最长序列 × 条数）")
    r.add_argument("--max-batch", type=int, default=32)
    r.add_argument("--limit", type=int, default=0, help="只测随机抽取的 n 题（冒烟用）")
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True, help="对照（基座）的逐题结果")
    c.add_argument("--a-name", default="base")
    c.add_argument("--b", nargs="+", required=True, help="name=path，与对照比较的模型")
    c.add_argument("--output", required=True)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
