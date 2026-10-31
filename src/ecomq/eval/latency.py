"""D4（DEVLOG S26）：部署时的速度。比较教师（7B）与学生（1.5B / 3B）在两种推理引擎、两种打分方式下的吞吐和延迟。

两种打分方式：
- full：评测一直用的方式，4 个候选回复各算一次完整的 log p（score_labels），一条样本 = 4 条序列；
- first：4 个候选回复的 token 序列在第一个不同的 token 之前完全相同（公共前缀，例如 {"relevance": "），
  把公共前缀接在提示后面，只做一次前向，比较 4 个标签各自第一个 token 的概率。一条样本 = 1 条序列。
  要求 4 个标签的第一个不同 token 互不相同（运行时断言）。与 full 的预测一致率一并报告。
两种引擎：
- hf：transformers（可加 4bit 与 LoRA），即目前评测用的实现；
- vllm：bf16 基座 + LoRA 适配器（vLLM 的多 LoRA 推理，不需要先合并成完整模型、不额外占磁盘），
  只支持 first 方式（取下一个 token 的 top-20 对数概率）。QLoRA 适配器是在 4bit 基座上训练的，
  在 bf16 基座上推理与训练时的数值不完全一致（P8），所以与 hf 4bit 的预测一致率要一并核对。
指标：吞吐（条/秒，一次提交全部样本）；单条延迟（逐条提交，p50 / p95，毫秒）。
输出：<output>.json（指标）与 <output>.preds.jsonl（每条样本各方式的预测，用于一致率）。
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.schema import RELEVANCE, build_user_prompt, format_output


def label_token_layout(tok, prompt_ids: list[int] | None = None) -> tuple[list[int], list[int]]:
    """返回 (公共前缀 token, 每个标签第一个不同的 token)。在一个具体提示后面分词，避免与提示末尾合并。"""
    from ecomq.eval.score_labels_cot import common_prefix_len

    seqs = [tok(format_output({"relevance": lab}, "relevance"), add_special_tokens=False)["input_ids"]
            for lab in RELEVANCE]
    k = common_prefix_len(seqs)
    first = [s[k] for s in seqs]
    assert len(set(first)) == len(first), f"labels share their first divergent token: {first}"
    return seqs[0][:k], first


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def run_hf(args, recs, prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from ecomq.eval.score_labels import label_logps

    tok = AutoTokenizer.from_pretrained(args.model)
    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    prefix, first = label_token_layout(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = [tok(p, add_special_tokens=False)["input_ids"] + prefix for p in prompts]

    def first_token(batch_idx: list[int]) -> list[list[float]]:
        width = max(len(seqs[i]) for i in batch_idx)
        ids = torch.full((len(batch_idx), width), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch_idx), width), dtype=torch.long)
        for j, i in enumerate(batch_idx):
            ids[j, : len(seqs[i])] = torch.tensor(seqs[i])
            att[j, : len(seqs[i])] = 1
        with torch.no_grad():
            logits = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).logits
        out = []
        for j, i in enumerate(batch_idx):
            lp = torch.log_softmax(logits[j, len(seqs[i]) - 1].float(), dim=-1)
            out.append([lp[t].item() for t in first])
        return out

    torch.cuda.synchronize()
    label_logps(model, tok, prompts[:64], args.batch_size)  # 预热
    t = time.time()
    full = label_logps(model, tok, prompts, args.batch_size)
    torch.cuda.synchronize()
    t_full = time.time() - t
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    t = time.time()
    fst = [None] * len(seqs)
    for s in range(0, len(order), args.batch_size):
        b = order[s : s + args.batch_size]
        for i, v in zip(b, first_token(b)):
            fst[i] = v
    torch.cuda.synchronize()
    t_first = time.time() - t
    lat = []
    for i in range(min(args.latency_n, len(seqs))):
        t = time.time()
        first_token([i])
        torch.cuda.synchronize()
        lat.append((time.time() - t) * 1000)
    preds = [{"id": r["id"], "full": max(full[i], key=full[i].get),
              "first": RELEVANCE[max(range(4), key=lambda c: fst[i][c])]} for i, r in enumerate(recs)]
    n = len(recs)
    return preds, {
        "full_rows_per_s": round(n / t_full, 2), "first_rows_per_s": round(n / t_first, 2),
        "first_latency_ms_p50": round(percentile(lat, 0.5), 1), "first_latency_ms_p95": round(percentile(lat, 0.95), 1),
        "first_vs_full_agreement": round(sum(p["full"] == p["first"] for p in preds) / n, 4),
        "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def run_vllm(args, recs, prompts):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.model)
    prefix, first = label_token_layout(tok)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_len,
              gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True,
              enable_lora=bool(args.lora), max_lora_rank=64)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    lora = LoRARequest("adapter", 1, args.lora) if args.lora else None
    inputs = [TokensPrompt(prompt_token_ids=tok(p, add_special_tokens=False)["input_ids"] + prefix) for p in prompts]

    def pick(o) -> str:
        lp = o.outputs[0].logprobs[0]
        vals = [lp[t].logprob if t in lp else float("-inf") for t in first]
        return RELEVANCE[max(range(4), key=lambda c: vals[c])]

    llm.generate(inputs[:64], sp, use_tqdm=False, lora_request=lora)  # 预热
    t = time.time()
    outs = llm.generate(inputs, sp, use_tqdm=False, lora_request=lora)
    t_first = time.time() - t
    lat = []
    for i in range(min(args.latency_n, len(inputs))):
        t = time.time()
        llm.generate([inputs[i]], sp, use_tqdm=False, lora_request=lora)
        lat.append((time.time() - t) * 1000)
    preds = [{"id": r["id"], "first": pick(o)} for r, o in zip(recs, outs)]
    return preds, {"first_rows_per_s": round(len(recs) / t_first, 2),
                   "first_latency_ms_p50": round(percentile(lat, 0.5), 1),
                   "first_latency_ms_p95": round(percentile(lat, 0.95), 1),
                   "labels_missing_from_top20": sum(all(t not in o.outputs[0].logprobs[0] for t in first) for o in outs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "vllm"], required=True)
    ap.add_argument("--model", required=True, help="基座目录（两种引擎都可加 --lora）")
    ap.add_argument("--lora")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--test", required=True)
    ap.add_argument("--n", type=int, default=2000, help="取测试集前 n 条")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output", required=True, help="输出前缀")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    recs = list(read_jsonl(args.test))[: args.n]
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": build_user_prompt(r["query"], r.get("title"), "relevance", r.get("product"))}],
        tokenize=False, add_generation_prompt=True) for r in recs]
    preds, res = (run_hf if args.engine == "hf" else run_vllm)(args, recs, prompts)
    res = {"name": args.name, "engine": args.engine, "model": Path(args.model).name, "n": len(recs),
           "load_4bit": args.load_4bit, "lora": bool(args.lora), **res}
    write_jsonl(args.output + ".preds.jsonl", preds)
    Path(args.output + ".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
