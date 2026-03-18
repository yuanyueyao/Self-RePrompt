"""
SRP 对比 Baseline 评测脚本。

对同一数据集评测以下 5 种条件，帮助定位 SRP 的增益来源：

  B0  Base Qwen3-8B（无任何修改）
  B1  Base + CoT prompt（零样本，在 user 消息中加策略提示）
  B2  Base + Oracle SRP（推理时把 teacher 生成的 sr_prompt 注入 user 消息）
  B3  Base + SRP-LoRA（我们的方法，无 SRP 格式加成，即 disable_adapter + 正常 prompt）
  B4  SRP-LoRA（我们的方法，enable_adapter）

B2 需要数据文件中存在 sr_prompt 字段（仅 srp_prompt_with_answer 目录下的文件有）。

用法：
    # 全部 5 种条件一次评测（会顺序评测，较慢）：
    CUDA_VISIBLE_DEVICES=0 python -u src/student/eval_baselines.py \\
        --dataset gsm8k --max_samples 200

    # 只评测指定 baseline（逗号分隔）：
    CUDA_VISIBLE_DEVICES=0 python -u src/student/eval_baselines.py \\
        --dataset hotpot --max_samples 200 --modes B0,B1,B2,B4

    # 使用 v3 adapter（默认）：
    CUDA_VISIBLE_DEVICES=0 python -u src/student/eval_baselines.py \\
        --dataset gsm8k --lora_dir outputs/qwen3_sr_lora_v3 --max_samples 200
"""

import argparse
import json
import re
import sys
import time
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ── CoT/策略提示模板 ─────────────────────────────────────────────

COT_SUFFIX = (
    "\n\nBefore answering, briefly identify the key reasoning strategy needed, "
    "then give your final answer."
)

ORACLE_SRP_TEMPLATE = (
    "{user}\n\n"
    "[Hint: Here is a strategy you can follow to answer this question]\n"
    "{sr_prompt}"
)

DATASET_CONFIGS = {
    "hotpot": {
        "file": "data/srp_prompt_with_answer/hotpot_train_qa_2000_with_srp_answer.jsonl",
        "user_key": "user",
        "gold_key": "answer",
        "srp_key": "sr_prompt",
        "type": "string",
    },
    "gsm8k": {
        "file": "data/srp_prompt_with_answer/gsm8k_train_with_srp_answer.jsonl",
        "user_key": "user",
        "gold_key": "answer",
        "srp_key": "sr_prompt",
        "type": "numeric",
    },
    "openbookqa": {
        "file": "data/srp_prompt_with_answer/openbookqa_train_with_srp_answer.jsonl",
        "user_key": "user",
        "gold_key": "answer",
        "srp_key": "sr_prompt",
        "type": "choice",
    },
}


def log(msg: str = ""):
    print(msg, flush=True)


# ── 答案提取（复用 eval_lora_accuracy 的逻辑）────────────────────

def extract_gsm8k_answer(text: str) -> str:
    text = re.split(r"<\|im_end\|>", text)[0].strip()
    num_pattern = r"-?\d+\.?\d*"

    def clean(s): return s.replace(",", "").strip()
    def first_num(s):
        m = re.search(num_pattern, clean(s))
        return m.group(0) if m else ""

    m = re.search(r"####\s*([^\*\n][^\n]*)", text)
    if m:
        n = first_num(m.group(1))
        if n: return n

    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        n = first_num(boxed.group(1))
        if n: return n

    tail = text[-600:]
    for marker in ["Final Answer", "Final answer", "Answer:", "answer:"]:
        idx = tail.rfind(marker)
        if idx >= 0:
            window = tail[idx + len(marker): idx + len(marker) + 200]
            n = first_num(window)
            if n: return n

    bold_nums = re.findall(r"\*\*([^*]+)\*\*", text)
    for b in reversed(bold_nums):
        n = first_num(b)
        if n: return n

    tail_short = text[-300:]
    numbers = re.findall(num_pattern, clean(tail_short))
    return numbers[-1] if numbers else ""


def extract_hotpot_answer(text: str) -> str:
    text = re.split(r"<\|im_end\|>", text)[0].strip()
    for prefix in ["the answer is", "answer:", "final answer:"]:
        idx = text.lower().rfind(prefix)
        if idx >= 0:
            return text[idx + len(prefix):].strip().rstrip(".").strip()
    return text.strip().split("\n")[-1].strip().rstrip(".").strip()


def extract_choice(text: str) -> str:
    text = re.split(r"<\|im_end\|>", text)[0].strip()
    for m in reversed(list(re.finditer(
        r"(?:answer|choice|correct)[\s\w]*?[:\-]?\s*\(?([A-D])\)?(?![A-Za-z])", text
    ))):
        return m.group(1).upper()
    m = re.search(r"\*\*([A-D])\*\*|\(([A-D])\)", text)
    if m:
        return (m.group(1) or m.group(2)).upper()
    matches = re.findall(r"\b([A-D])\b", text)
    return matches[-1].upper() if matches else ""


def match_answer(pred: str, gold: str, dtype: str) -> bool:
    if dtype == "numeric":
        p = extract_gsm8k_answer(pred).replace(",", "")
        g = gold.strip().replace(",", "")
        if not p or not g: return False
        try: return float(p) == float(g)
        except ValueError: return p == g
    elif dtype == "choice":
        p = extract_choice(pred)
        m = re.match(r"([A-D])", gold.strip())
        return bool(p and m and p == m.group(1).upper())
    else:
        p = extract_hotpot_answer(pred).lower()
        g = gold.strip().lower()
        return g in p or p in g


# ── 数据加载 ─────────────────────────────────────────────────────

def load_data(cfg: dict, max_samples: int, seed: int) -> list:
    data = []
    with open(cfg["file"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            user = (obj.get(cfg["user_key"]) or "").strip()
            gold = (obj.get(cfg["gold_key"]) or "").strip()
            sr_prompt = (obj.get(cfg.get("srp_key", "sr_prompt")) or "").strip()
            if user and gold:
                data.append({"user": user, "gold": gold, "sr_prompt": sr_prompt})
    random.seed(seed)
    random.shuffle(data)
    return data[:max_samples]


# ── 推理 ─────────────────────────────────────────────────────────

def build_prompt(tokenizer, user_content: str) -> str:
    msgs = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, user_content: str,
             max_new_tokens: int, device: str) -> str:
    prompt = build_prompt(tokenizer, user_content)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for tok in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=list(stop_ids) or tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs.input_ids.shape[1]:]
    if len(generated) >= max_new_tokens:
        log(f"  ⚡ 警告：达到 max_new_tokens={max_new_tokens} 上限！")
    return tokenizer.decode(generated, skip_special_tokens=False)


# ── 单模式评测 ──────────────────────────────────────────────────

def eval_one_mode(mode: str, data: list, model, tokenizer,
                  dtype: str, max_new_tokens: int, device: str) -> dict:
    """
    评测单个 baseline 模式，返回 {acc, n, corrects}。
    mode 含义：
      B0  base model 原始输入
      B1  base model + CoT suffix
      B2  base model + oracle sr_prompt 注入 user 消息
      B3  base model (disable adapter) + 原始输入（与 B0 相同，用于确认 LoRA 权重差异）
      B4  LoRA model（enable adapter）+ 原始输入
    """
    if mode in ("B0", "B3"):
        model.disable_adapter_layers()
    elif mode == "B4":
        model.enable_adapter_layers()
    elif mode in ("B1", "B2"):
        model.disable_adapter_layers()

    corrects = []
    for i, sample in enumerate(data):
        user = sample["user"]
        gold = sample["gold"]

        if mode == "B1":
            user_input = user + COT_SUFFIX
        elif mode == "B2":
            sr = sample.get("sr_prompt", "").strip()
            if sr:
                user_input = ORACLE_SRP_TEMPLATE.format(user=user, sr_prompt=sr)
            else:
                user_input = user  # 无 sr_prompt 时退化到 B0
        else:
            user_input = user

        resp = generate(model, tokenizer, user_input, max_new_tokens, device)
        ok = match_answer(resp, gold, dtype)
        corrects.append(ok)

        # 实时进度
        n = i + 1
        acc = sum(corrects) / n * 100
        log(f"  [{mode}] [{n:3d}/{len(data)}] {'✓' if ok else '✗'}  acc={acc:5.1f}%")

    model.enable_adapter_layers()  # 恢复默认状态
    return {"mode": mode, "n": len(data), "acc": sum(corrects) / len(data) * 100,
            "corrects": corrects}


# ── main ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SRP baseline comparison")
    p.add_argument("--base_model", default="model/Qwen3-8B")
    p.add_argument("--lora_dir", default="outputs/qwen3_sr_lora_v3")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="gsm8k")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modes", type=str, default="B0,B1,B2,B4",
                   help="要评测的模式，逗号分隔。可选：B0,B1,B2,B3,B4")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    log(f"加载数据：{cfg['file']}（最多 {args.max_samples} 条）")
    data = load_data(cfg, args.max_samples, args.seed)
    log(f"实际样本数：{len(data)}\n")

    # B2 需要 sr_prompt，统计覆盖率
    if "B2" in modes:
        has_srp = sum(1 for d in data if d.get("sr_prompt"))
        log(f"Oracle SRP 覆盖率：{has_srp}/{len(data)} ({has_srp/len(data)*100:.0f}%)")
        if has_srp == 0:
            log("  ⚠️  数据中无 sr_prompt，跳过 B2")
            modes = [m for m in modes if m != "B2"]

    log(f"\n加载模型：{args.base_model} + {args.lora_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    lora_tok_cfg = Path(args.lora_dir) / "tokenizer_config.json"
    if lora_tok_cfg.exists():
        lora_cfg = json.loads(lora_tok_cfg.read_text(encoding="utf-8"))
        extra = lora_cfg.get("extra_special_tokens", [])
        if extra:
            tokenizer.add_special_tokens({"additional_special_tokens": extra})
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()
    log("模型加载完成\n")

    all_results = []
    t_start = time.time()

    for mode in modes:
        mode_desc = {
            "B0": "Base Qwen3-8B（无修改）",
            "B1": "Base + CoT/策略提示（零样本）",
            "B2": "Base + Oracle SRP（teacher sr_prompt 注入）",
            "B3": "Base（disable adapter，验证等价于 B0）",
            "B4": "SRP-LoRA v3（我们的方法）",
        }
        log(f"\n{'─'*60}")
        log(f"  评测模式 {mode}：{mode_desc.get(mode, mode)}")
        log(f"{'─'*60}")
        t0 = time.time()
        result = eval_one_mode(mode, data, model, tokenizer,
                               cfg["type"], args.max_new_tokens, args.device)
        result["time"] = time.time() - t0
        all_results.append(result)

    # ── 汇总表格 ──────────────────────────────────────────────
    total_time = time.time() - t_start
    n = len(data)

    log(f"\n{'='*65}")
    log(f"  数据集: {args.dataset}  |  样本数: {n}  |  总耗时: {total_time:.0f}s")
    log(f"  LoRA:  {args.lora_dir}")
    log(f"{'='*65}")
    log(f"  {'模式':<6}  {'准确率':>8}  {'正确':>6}  {'描述'}")
    log(f"  {'─'*60}")

    for r in all_results:
        mode_desc_short = {
            "B0": "Base（无修改）",
            "B1": "Base + CoT prompt（zero-shot）",
            "B2": "Base + Oracle SRP（teacher hint）",
            "B3": "Base（disable_adapter，同 B0）",
            "B4": "SRP-LoRA（our method）",
        }
        log(f"  {r['mode']:<6}  {r['acc']:>7.1f}%  {round(r['acc']*n/100):>5}/{n}  "
            f"{mode_desc_short.get(r['mode'], r['mode'])}")

    # B4 vs 各 baseline 增益
    b4 = next((r for r in all_results if r["mode"] == "B4"), None)
    if b4:
        log(f"\n  SRP-LoRA vs 各 Baseline：")
        for r in all_results:
            if r["mode"] == "B4": continue
            delta = b4["acc"] - r["acc"]
            sign = "🟢" if delta > 0 else ("🔴" if delta < 0 else "➡️")
            log(f"    vs {r['mode']}:  {delta:+.1f}%  {sign}")

    log(f"{'='*65}")


if __name__ == "__main__":
    main()
