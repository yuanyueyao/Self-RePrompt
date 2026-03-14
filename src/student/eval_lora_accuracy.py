"""
评测 LoRA adapter 对模型准确率的影响。

对比 base model（禁用 LoRA）和 base + LoRA 在数据集上的准确率，
检测 SRP 格式是否被正确触发，以及最终答案质量的变化。

支持数据集：HotpotQA（字符串匹配）、GSM8K（数字匹配）、OpenBookQA（选项匹配）。

用法：
    python src/student/eval_lora_accuracy.py \
        --lora_dir outputs/qwen3_sr_lora_hotpot_v2 \
        --dataset hotpot \
        --max_samples 100

    python src/student/eval_lora_accuracy.py \
        --lora_dir outputs/qwen3_sr_lora_hotpot_v2 \
        --dataset gsm8k \
        --max_samples 100
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def log(msg: str = ""):
    print(msg, flush=True)


DATASET_CONFIGS = {
    "hotpot": {
        "file": "data/srp_prompt_with_answer/hotpot_train_qa_2000_with_srp_answer.jsonl",
        "user_key": "user",
        "gold_key": "answer",
        "type": "string",
    },
    "gsm8k": {
        "file": "data/srp_prompt_with_answer/gsm8k_train_with_srp_answer.jsonl",
        "user_key": "user",
        "gold_key": "answer",
        "type": "numeric",
    },
    "openbookqa": {
        "file": "data/srp_prompt_with_answer/openbookqa_train_with_srp_answer.jsonl",
        "user_key": "user",
        "gold_key": "answer",
        "type": "choice",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="model/Qwen3-8B")
    p.add_argument("--lora_dir", default="outputs/qwen3_sr_lora_hotpot_v2")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="hotpot")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42, help="随机采样 seed")
    return p.parse_args()


# ── 答案提取 ────────────────────────────────────────────────

def extract_hotpot_answer(text: str) -> str:
    text = text.strip()
    for prefix in ["the answer is", "answer:", "final answer:"]:
        idx = text.lower().rfind(prefix)
        if idx >= 0:
            return text[idx + len(prefix):].strip().rstrip(".").strip()
    lines = text.strip().split("\n")
    return lines[-1].strip().rstrip(".").strip()


def extract_gsm8k_answer(text: str) -> str:
    text = text.strip()
    if "####" in text:
        after = text.split("####")[-1].strip()
        m = re.search(r"-?[\d,]+\.?\d*", after.replace(",", ""))
        if m:
            return m.group(0)
    boxed = re.search(r"\\boxed\{[^}]*?(-?[\d,]+\.?\d*)", text)
    if boxed:
        return boxed.group(1).replace(",", "")
    tail = text[-400:] if len(text) > 400 else text
    for marker in ["Final Answer", "Final answer", "Answer:", "answer:", "**"]:
        if marker in tail:
            idx = tail.rfind(marker)
            nums = re.findall(r"-?[\d,]+\.?\d*", tail[idx:].replace(",", ""))
            if nums:
                return nums[-1]
    tail = text[-300:] if len(text) > 300 else text
    numbers = re.findall(r"-?[\d,]+\.?\d*", tail.replace(",", ""))
    return numbers[-1] if numbers else ""


def extract_choice(text: str) -> str:
    text = text.strip()
    for m in reversed(list(re.finditer(
        r"(?:answer|choice|correct)[\s\w]*?[:\-]?\s*\(?([A-D])\)?(?![A-Za-z])",
        text
    ))):
        return m.group(1).upper()
    m = re.search(r"\*\*([A-D])\*\*|\(([A-D])\)", text)
    if m:
        return (m.group(1) or m.group(2)).upper()
    matches = re.findall(r"\b([A-D])\b", text)
    return matches[-1].upper() if matches else ""


def match_answer(pred_text: str, gold: str, dtype: str) -> bool:
    if dtype == "numeric":
        pred = extract_gsm8k_answer(pred_text).replace(",", "")
        gold = gold.strip().replace(",", "")
        if not pred or not gold:
            return False
        try:
            return float(pred) == float(gold)
        except ValueError:
            return pred == gold
    elif dtype == "choice":
        pred = extract_choice(pred_text)
        gold_letter = re.match(r"([A-D])", gold.strip())
        return bool(pred and gold_letter and pred == gold_letter.group(1).upper())
    else:
        pred = extract_hotpot_answer(pred_text).lower()
        gold_lower = gold.strip().lower()
        return gold_lower in pred or pred in gold_lower


# ── 数据加载 ────────────────────────────────────────────────

def load_data(config: dict, max_samples: int, seed: int):
    import random
    data = []
    with open(config["file"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            user = (obj.get(config["user_key"]) or "").strip()
            gold = (obj.get(config["gold_key"]) or "").strip()
            if user and gold:
                data.append({"user": user, "gold": gold})
    random.seed(seed)
    random.shuffle(data)
    return data[:max_samples]


# ── 推理 ────────────────────────────────────────────────────

def generate(model, tokenizer, user: str, max_new_tokens: int, device: str) -> str:
    msgs = [{"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)


# ── main ────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]

    log(f"加载数据：{cfg['file']}（最多 {args.max_samples} 条）")
    data = load_data(cfg, args.max_samples, args.seed)
    log(f"实际样本数：{len(data)}")

    log(f"\n加载模型：{args.base_model} + {args.lora_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()
    log("模型加载完成\n")

    results = {"base": [], "lora": []}
    srp_format_count = 0
    t_start = time.time()

    for i, sample in enumerate(data):
        user = sample["user"]
        gold = sample["gold"]

        model.enable_adapter_layers()
        t0 = time.time()
        resp_lora = generate(model, tokenizer, user, args.max_new_tokens, args.device)
        t_lora = time.time() - t0

        model.disable_adapter_layers()
        t0 = time.time()
        resp_base = generate(model, tokenizer, user, args.max_new_tokens, args.device)
        t_base = time.time() - t0

        ok_lora = match_answer(resp_lora, gold, cfg["type"])
        ok_base = match_answer(resp_base, gold, cfg["type"])
        has_srp = "<SRP_START>" in resp_lora and "<SRP_END>" in resp_lora

        results["lora"].append(ok_lora)
        results["base"].append(ok_base)
        if has_srp:
            srp_format_count += 1

        n = i + 1
        acc_l = sum(results["lora"]) / n * 100
        acc_b = sum(results["base"]) / n * 100
        srp_r = srp_format_count / n * 100
        elapsed = time.time() - t_start
        eta = elapsed / n * (len(data) - n)

        mark_l = "✓" if ok_lora else "✗"
        mark_b = "✓" if ok_base else "✗"
        srp_flag = "SRP" if has_srp else "---"

        log(f"\n{'─'*70}")
        log(
            f"[{n:3d}/{len(data)}] base={mark_b} lora={mark_l} {srp_flag} "
            f"| acc base={acc_b:5.1f}% lora={acc_l:5.1f}% Δ={acc_l-acc_b:+.1f}% "
            f"| {t_lora:.1f}s+{t_base:.1f}s  ETA {eta:.0f}s"
        )
        log(f"  Q:    {user[:200]}{'...' if len(user)>200 else ''}")
        log(f"  Gold: {gold}")
        log(f"  ── Base 输出 ──")
        log(f"  {resp_base.strip()[:500]}")
        log(f"  ── LoRA 输出 ──")
        log(f"  {resp_lora.strip()[:500]}")
        if ok_lora != ok_base:
            tag = "  >>> 🟢 LoRA纠正" if ok_lora else "  >>> ⚠️  LoRA误导"
            log(tag)

    # ── 最终统计 ──
    n = len(data)
    acc_base = sum(results["base"]) / n * 100
    acc_lora = sum(results["lora"]) / n * 100
    srp_rate = srp_format_count / n * 100
    total_time = time.time() - t_start

    both_correct = sum(b and l for b, l in zip(results["base"], results["lora"]))
    both_wrong = sum(not b and not l for b, l in zip(results["base"], results["lora"]))
    corrected = sum(not b and l for b, l in zip(results["base"], results["lora"]))
    misleading = sum(b and not l for b, l in zip(results["base"], results["lora"]))

    log("\n" + "=" * 60)
    log(f"  数据集: {args.dataset}  |  样本数: {n}  |  耗时: {total_time:.0f}s")
    log(f"  LoRA:  {args.lora_dir}")
    log("=" * 60)
    log(f"  Base 准确率:   {acc_base:5.1f}%  ({sum(results['base'])}/{n})")
    log(f"  LoRA 准确率:   {acc_lora:5.1f}%  ({sum(results['lora'])}/{n})")
    log(f"  净增益:        {acc_lora - acc_base:+.1f}%  ({corrected - misleading:+d} 条)")
    log(f"  SRP 格式触发率: {srp_rate:.0f}%")
    log()
    log(f"  四象限:")
    log(f"    both_correct : {both_correct:3d}  ({both_correct/n*100:5.1f}%)")
    log(f"    corrected    : {corrected:3d}  ({corrected/n*100:5.1f}%)  🟢")
    log(f"    both_wrong   : {both_wrong:3d}  ({both_wrong/n*100:5.1f}%)")
    log(f"    misleading   : {misleading:3d}  ({misleading/n*100:5.1f}%)  ⚠️")
    log("=" * 60)

    model.enable_adapter_layers()


if __name__ == "__main__":
    main()
