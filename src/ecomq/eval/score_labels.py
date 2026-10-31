"""对 relevance 任务的每条样本，计算 4 个候选输出的对数概率 log p(输出 | 提示)。用于推理时先验调整（DEVLOG S12）。

候选输出与训练时的回复完全一致：format_output({"relevance": label}) 再加上对话模板的结束符
（Qwen 为 <|im_end|>）。整段回复的对数概率 = 回复中每个 token 的对数概率之和，
这样不同长度的标签之间可以直接比较（都是“生成完整回复并结束”的概率）。
使用右侧填充：Qwen2 前向时按位置序号计算位置编码，左侧填充会改变位置、影响 logits。
输出：每行 {"id", "logp": {label: float}}。
"""
import argparse

from ecomq.data.io import read_jsonl, write_jsonl
from ecomq.schema import LABELS, RELEVANCE, build_user_prompt, format_output, task_of


def label_logps(model, tok, prompts: list[str], batch_size: int = 32,
                labels: list[str] = RELEVANCE) -> list[dict[str, float]]:
    """prompts 是已套好对话模板的完整提示（以 assistant 开头结束）；返回每条提示下各候选回复的 log p。
    labels 默认是四分类的 4 个标签；relevance_zh（D5）传入 2 个标签。候选回复都是 {"relevance": label}。"""
    items = []  # (提示下标, 标签, 完整 token 序列, 回复起始位置)
    for i, prompt in enumerate(prompts):
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        for lab in labels:
            text = prompt + format_output({"relevance": lab}, "relevance") + tok.eos_token
            full = tok(text, add_special_tokens=False)["input_ids"]
            assert full[: len(p_ids)] == p_ids, "prompt tokens changed after appending the response"
            items.append((i, lab, full, len(p_ids)))
    return score_items(model, tok, items, len(prompts), batch_size)


def score_items(model, tok, items: list[tuple], n: int, batch_size: int) -> list[dict[str, float]]:
    """items 为 (提示下标, 标签, 完整 token 序列, 计分起始位置)；返回 n 个 {标签: 起始位置之后 token 的 log p 之和}。
    使用右侧填充：Qwen2 前向时按位置序号计算位置编码，左侧填充会改变位置、影响 logits。"""
    import torch

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(len(items)), key=lambda k: len(items[k][2]))  # 按长度分批，减少填充
    scores = [dict() for _ in range(n)]
    for s in range(0, len(order), batch_size):
        batch = [items[k] for k in order[s : s + batch_size]]
        width = max(len(b[2]) for b in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for j, b in enumerate(batch):
            ids[j, : len(b[2])] = torch.tensor(b[2])
            att[j, : len(b[2])] = 1
        with torch.no_grad():
            logits = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).logits
        for j, (i, lab, full, start) in enumerate(batch):
            # 位置 t 的 logits 预测第 t+1 个 token；计分的 token 位于 [start, len(full))
            lp = torch.log_softmax(logits[j, start - 1 : len(full) - 1].float(), dim=-1)
            target = ids[j, start : len(full)].to(lp.device)
            scores[i][lab] = lp.gather(-1, target.unsqueeze(-1)).sum().item()
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", help="LoRA 适配器目录；不给时直接用基座（零样本）")
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--task", default="relevance", choices=["relevance", "relevance_zh"])
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

    recs = list(read_jsonl(args.test))
    assert all(task_of(r) == args.task for r in recs)
    labels = LABELS[args.task]
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": build_user_prompt(r["query"], r.get("title"), args.task, r.get("product"))}],
        tokenize=False, add_generation_prompt=True) for r in recs]
    scores = label_logps(model, tok, prompts, args.batch_size, labels)
    write_jsonl(args.output, [{"id": r["id"], "logp": scores[i]} for i, r in enumerate(recs)])
    print(f"写出 {len(recs)} 条（{len(recs) * len(labels)} 个候选）-> {args.output}")


if __name__ == "__main__":
    main()
