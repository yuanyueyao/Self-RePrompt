"""
评测 LoRA adapter 对模型准确率的影响。

对比 base model（禁用 LoRA）和 base + LoRA 在数据集上的准确率，
检测 SRP 格式是否被正确触发，以及最终答案质量的变化。

支持数据集：HotpotQA（字符串匹配）、GSM8K（数字匹配）、OpenBookQA（选项匹配）、MATH（LaTeX 答案匹配）。

用法：
    python src/student/eval_lora_accuracy.py \
        --lora_dir outputs/qwen3_sr_lora_v3 \
        --dataset math \
        --max_samples 200

    python src/student/eval_lora_accuracy.py \
        --lora_dir outputs/qwen3_sr_lora_v3 \
        --dataset math \
        --math_subject algebra \
        --math_level "Level 3,Level 4,Level 5" \
        --max_samples 200
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
    "math": {
        "file": "data/raw/hendrycks_math.json",
        "user_key": "question",
        "gold_key": "answer",
        "type": "math",
        "split": "test",          # 用 test split（5000 条）
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="model/Qwen3-8B")
    p.add_argument("--lora_dir", default="outputs/qwen3_sr_lora_v3")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="hotpot")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=4096,
                   help="硬上限，正常情况下模型会在停止符处提前结束，不会跑满这个值")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42, help="随机采样 seed")
    # MATH 专用过滤参数
    p.add_argument("--math_subject", type=str, default="",
                   help="逗号分隔的 subject 过滤，如 'algebra,geometry'，空则全部")
    p.add_argument("--math_level", type=str, default="",
                   help="逗号分隔的 level 过滤，如 'Level 1,Level 2,Level 3'，空则全部")
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
    num_pattern = r"-?\d+\.?\d*"

    def clean_num(s: str) -> str:
        return s.replace(",", "").strip()

    def first_num(s: str) -> str:
        """在字符串 s 中找第一个数字（去掉逗号后），失败返回空串。"""
        m = re.search(num_pattern, clean_num(s))
        return m.group(0) if m else ""

    # 0. 预处理：去掉 <|im_end|> 及其之后的所有内容（含尾部闲聊）
    text = re.split(r"<\|im_end\|>", text)[0].rstrip()

    # 1. GSM8K 官方 #### 格式：#### 后紧跟空白+数字，而非 markdown 标题（#### **...）
    gsm_marker = re.search(r"####\s*([^\*\n][^\n]*)", text)
    if gsm_marker:
        n = first_num(gsm_marker.group(1))
        if n:
            return n

    # 2. LaTeX \boxed{}
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        n = first_num(boxed.group(1))
        if n:
            return n

    # 3. 明确的答案 marker：只在 marker 之后 200 字符内取 **第一个** 数字
    #    避免 "the answer is 7800\n\nLet me know about the 6 years..." 这类干扰
    tail = text[-600:]
    for marker in ["Final Answer", "Final answer", "Answer:", "answer:"]:
        idx = tail.rfind(marker)
        if idx >= 0:
            window = tail[idx + len(marker): idx + len(marker) + 200]
            n = first_num(window)
            if n:
                return n

    # 4. 加粗数字：** ... ** 模式（markdown 强调答案），取加粗内容里的第一个数字
    bold_nums = re.findall(r"\*\*([^*]+)\*\*", text)
    for b in reversed(bold_nums):   # 从后向前，优先取最后一个加粗数字
        n = first_num(b)
        if n:
            return n

    # 5. 最后兜底：取文本末尾 300 字符中最后一个数字
    tail_short = text[-300:]
    numbers = re.findall(num_pattern, clean_num(tail_short))
    return numbers[-1] if numbers else ""


def extract_boxed(text: str) -> str:
    """
    从文本中提取最后一个 \\boxed{...} 的内容，支持嵌套花括号。
    例如 \\boxed{\\dfrac{9}{4}} → '\\dfrac{9}{4}'
    """
    text = re.split(r"<\|im_end\|>", text)[0]
    # 找到所有 \boxed{ 的起始位置，取最后一个
    starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return ""
    pos = starts[-1] + len(r"\boxed{")
    depth = 1
    buf = []
    while pos < len(text) and depth > 0:
        c = text[pos]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(c)
        pos += 1
    return "".join(buf).strip()


def normalize_math(s: str) -> str:
    """规范化 LaTeX 答案字符串，用于比较。"""
    s = s.strip()
    # 去掉展示命令：\dfrac → \frac，\left/\right → 空
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = re.sub(r"\\left|\\right", "", s)
    # 去掉多余空白
    s = re.sub(r"\s+", "", s)
    # 去掉美元符
    s = s.replace("$", "")
    return s.lower()


def match_math(pred_text: str, gold: str) -> bool:
    """比较 MATH 数据集答案。先提取 boxed，再多策略比较。"""
    pred = extract_boxed(pred_text)
    if not pred:
        # fallback：取最后一个数字
        nums = re.findall(r"-?\d+\.?\d*", pred_text.replace(",", ""))
        pred = nums[-1] if nums else ""
    if not pred or not gold:
        return False

    # 1. 规范化字符串完全匹配
    if normalize_math(pred) == normalize_math(gold):
        return True

    # 2. 数值匹配（都能转 float）
    try:
        pv = float(pred.replace(",", ""))
        gv = float(gold.replace(",", ""))
        return abs(pv - gv) < 1e-6
    except (ValueError, TypeError):
        pass

    # 3. 去掉换行和空格后原始字符串匹配
    return pred.replace(" ", "") == gold.replace(" ", "")


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
    elif dtype == "math":
        return match_math(pred_text, gold)
    else:
        pred = extract_hotpot_answer(pred_text).lower()
        gold_lower = gold.strip().lower()
        return gold_lower in pred or pred in gold_lower


# ── 数据加载 ────────────────────────────────────────────────

def load_data(config: dict, max_samples: int, seed: int, args=None):
    import random
    data = []
    fpath = config["file"]

    if fpath.endswith(".json"):
        # JSON 格式（MATH）：可能是 {"train": [...], "test": [...]} 或直接 list
        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        split = config.get("split", "train")
        rows = raw[split] if isinstance(raw, dict) else raw

        # MATH 专用过滤
        subjects = {s.strip() for s in (getattr(args, "math_subject", "") or "").split(",") if s.strip()}
        levels   = {l.strip() for l in (getattr(args, "math_level",   "") or "").split(",") if l.strip()}
        for obj in rows:
            if subjects and obj.get("subject", "") not in subjects:
                continue
            if levels and obj.get("level", "") not in levels:
                continue
            user = (obj.get(config["user_key"]) or "").strip()
            gold = (obj.get(config["gold_key"]) or "").strip()
            if user and gold:
                data.append({"user": user, "gold": gold,
                             "meta": f"{obj.get('subject','')} / {obj.get('level','')}"})
    else:
        # JSONL 格式
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                user = (obj.get(config["user_key"]) or "").strip()
                gold = (obj.get(config["gold_key"]) or "").strip()
                if user and gold:
                    data.append({"user": user, "gold": gold, "meta": ""})

    random.seed(seed)
    random.shuffle(data)
    return data[:max_samples]


# ── 推理 ────────────────────────────────────────────────────

def generate(model, tokenizer, user: str, max_new_tokens: int, device: str,
             enable_thinking: bool = False) -> str:
    msgs = [{"role": "user", "content": user}]
    # enable_thinking=False：禁用 Qwen3 的 CoT thinking 模式，
    # 避免 base 模型生成超长 <think> 链导致被 max_new_tokens 截断、答案无法提取。
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # 旧版 tokenizer 不支持 enable_thinking 参数，降级处理
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 收集所有可能的停止 token id：eos_token、<|im_end|> 等
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for tok in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    eos_token_id = list(stop_ids) if stop_ids else tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = out[0][inputs.input_ids.shape[1]:]
    # 检查是否因 max_new_tokens 截断（未出现任何停止 token）
    if len(generated) >= max_new_tokens:
        log(f"  ⚡ 警告：输出达到 max_new_tokens={max_new_tokens} 上限，可能被截断！")
    return tokenizer.decode(generated, skip_special_tokens=False)


# ── main ────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]

    log(f"加载数据：{cfg['file']}（最多 {args.max_samples} 条）")
    data = load_data(cfg, args.max_samples, args.seed, args)
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
        # LoRA 模型已被训练为跳过 thinking、直接输出 SRP+答案，保持原始行为
        resp_lora = generate(model, tokenizer, user, args.max_new_tokens, args.device,
                             enable_thinking=False)
        t_lora = time.time() - t0

        model.disable_adapter_layers()
        t0 = time.time()
        # Base 模型禁用 thinking，避免超长 CoT 被 max_new_tokens 截断导致答案无法提取
        resp_base = generate(model, tokenizer, user, args.max_new_tokens, args.device,
                             enable_thinking=False)
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
        meta = sample.get("meta", "")
        log(f"  Q:    {'['+meta+'] ' if meta else ''}{user}")
        log(f"  Gold: {gold}")
        log(f"  ── Base 输出 ──")
        log(resp_base.strip())
        log(f"  ── LoRA 输出 ──")
        log(resp_lora.strip())
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
