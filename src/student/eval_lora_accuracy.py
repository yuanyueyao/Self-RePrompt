"""
评测 LoRA adapter 对模型准确率的影响。

─── 整体逻辑 ───────────────────────────────────────────────────
1. 数据加载（load_data）
   从指定数据集文件中随机采样 max_samples 条，统一转换为
   {user, gold, meta} 格式。支持 JSON（MATH/PopQA）和 JSONL
   （HotpotQA/GSM8K/OpenBookQA）两种格式。

2. 模型加载
   从 base_model 路径加载 Qwen3-8B，并挂载 LoRA adapter（PeftModel）。
   tokenizer 从 base 加载后追加 adapter 目录中声明的特殊 token
   （<SRP_START> / <SRP_END>），避免 tokenizer_config 格式兼容问题。

3. 逐样本推理（run_eval）
   对每条样本分别做两次推理：
   · enable_adapter  → LoRA 输出（含 SRP 格式）
   · disable_adapter → Base 输出（同一权重，禁用 LoRA）
   两次推理共享同一个已加载的模型实例，无需重复加载。

4. 答案匹配（match_answer）
   根据数据集类型选用不同策略：
   · string  (HotpotQA) — 关键词提取 + 子串包含
   · numeric (GSM8K)    — #### / boxed / bold 数字提取后数值比较
   · choice  (OBQA)     — 正则提取 A–D 选项字母
   · math    (MATH)     — \\boxed{} 提取 + LaTeX 规范化 + 数值比较
   · popqa   (PopQA)    — 在整段输出中搜索 possible_answers 任一子串

5. 汇总统计（_print_summary）
   报告 Base / LoRA 准确率、净增益、SRP 触发率及四象限
   （both_correct / corrected / both_wrong / misleading）。

─── 多卡并行 ───────────────────────────────────────────────────
通过 --gpus 0,1 启用多卡模式：
· 主进程将数据按 index % N 分片，为每张 GPU spawn 一个子进程
  （通过隐藏参数 --_split_id / --_total_splits / --_output_json 传递分片信息）。
· 子进程独立完成推理，将结果写入临时 JSON 文件。
· 主进程等待所有子进程结束后，调用 _merge_parts 合并分片结果并打印汇总。

─── 支持数据集 ─────────────────────────────────────────────────
  hotpot    HotpotQA      字符串匹配
  gsm8k     GSM8K         数字匹配
  openbookqa OpenBookQA   选项匹配（A–D）
  math      Hendrycks MATH  LaTeX 答案匹配
  popqa     PopQA         多候选答案子串匹配

─── 用法 ───────────────────────────────────────────────────────
单卡：
    python src/student/eval_lora_accuracy.py --dataset popqa --max_samples 200

多卡（自动并行，速度约快 N 倍）：
    python src/student/eval_lora_accuracy.py --dataset popqa --max_samples 200 --gpus 0,1

MATH 专用过滤：
    python src/student/eval_lora_accuracy.py --dataset math --max_samples 200 \\
        --math_subject algebra --math_level "Level 3,Level 4,Level 5"
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
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
        "split": "test",
    },
    "popqa": {
        "file": "data/raw/popqa.json",
        "user_key": "user",
        "gold_key": "possible_answers",
        "type": "popqa",
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
    p.add_argument("--gpus", type=str, default="",
                   help="逗号分隔的 GPU id，如 '0,1'。不填则自动使用 cuda:0。"
                        "填多张时自动多进程并行，速度约快 N 倍")
    p.add_argument("--seed", type=int, default=42, help="随机采样 seed")
    # MATH 专用过滤参数
    p.add_argument("--math_subject", type=str, default="",
                   help="逗号分隔的 subject 过滤，如 'algebra,geometry'，空则全部")
    p.add_argument("--math_level", type=str, default="",
                   help="逗号分隔的 level 过滤，如 'Level 1,Level 2,Level 3'，空则全部")
    # ── 以下参数为多卡模式内部使用，用户无需手动填写 ──
    p.add_argument("--_split_id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--_total_splits", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--_output_json", type=str, default="", help=argparse.SUPPRESS)
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
        m = re.search(num_pattern, clean_num(s))
        return m.group(0) if m else ""

    text = re.split(r"<\|im_end\|>", text)[0].rstrip()

    gsm_marker = re.search(r"####\s*([^\*\n][^\n]*)", text)
    if gsm_marker:
        n = first_num(gsm_marker.group(1))
        if n:
            return n

    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        n = first_num(boxed.group(1))
        if n:
            return n

    tail = text[-600:]
    for marker in ["Final Answer", "Final answer", "Answer:", "answer:"]:
        idx = tail.rfind(marker)
        if idx >= 0:
            window = tail[idx + len(marker): idx + len(marker) + 200]
            n = first_num(window)
            if n:
                return n

    bold_nums = re.findall(r"\*\*([^*]+)\*\*", text)
    for b in reversed(bold_nums):
        n = first_num(b)
        if n:
            return n

    tail_short = text[-300:]
    numbers = re.findall(num_pattern, clean_num(tail_short))
    return numbers[-1] if numbers else ""


def extract_boxed(text: str) -> str:
    """从文本中提取最后一个 \\boxed{...} 的内容，支持嵌套花括号。"""
    text = re.split(r"<\|im_end\|>", text)[0]
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
    s = s.strip()
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("$", "")
    return s.lower()


def match_math(pred_text: str, gold: str) -> bool:
    pred = extract_boxed(pred_text)
    if not pred:
        nums = re.findall(r"-?\d+\.?\d*", pred_text.replace(",", ""))
        pred = nums[-1] if nums else ""
    if not pred or not gold:
        return False
    if normalize_math(pred) == normalize_math(gold):
        return True
    try:
        pv = float(pred.replace(",", ""))
        gv = float(gold.replace(",", ""))
        return abs(pv - gv) < 1e-6
    except (ValueError, TypeError):
        pass
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


def match_popqa(pred_text: str, gold) -> bool:
    """
    PopQA: gold 为 possible_answers 的 JSON 字符串或 list，匹配任一即可。
    在整段输出中搜索，避免长输出时答案出现在中间被漏判。
    """
    text = re.split(r"<\|im_end\|>", pred_text)[0].strip().lower()
    if not text:
        return False
    try:
        possible = json.loads(gold) if isinstance(gold, str) else gold
    except (json.JSONDecodeError, TypeError):
        return False
    if not possible or not isinstance(possible, (list, tuple)):
        return False
    for ans in possible:
        ans_str = str(ans).strip().lower()
        if ans_str and ans_str in text:
            return True
    return False


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
    elif dtype == "popqa":
        return match_popqa(pred_text, gold)
    else:
        pred = extract_hotpot_answer(pred_text).lower()
        gold_lower = gold.strip().lower()
        return gold_lower in pred or pred in gold_lower


# ── 数据加载 ────────────────────────────────────────────────

def load_data(config: dict, max_samples: int, seed: int, args=None) -> list:
    data = []
    fpath = config["file"]

    if fpath.endswith(".json"):
        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        split = config.get("split", "train")
        rows = raw[split] if isinstance(raw, dict) else raw

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
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
    if len(generated) >= max_new_tokens:
        log(f"  ⚡ 警告：输出达到 max_new_tokens={max_new_tokens} 上限，可能被截断！")
    return tokenizer.decode(generated, skip_special_tokens=False)


# ── 结果合并（多卡） ──────────────────────────────────────────

def _merge_parts(parts: list) -> dict:
    """合并多卡分片结果，按原始 index 交错还原顺序。"""
    parts = sorted(parts, key=lambda x: x["split_id"])
    total_splits = len(parts)
    max_n = max(p["n"] for p in parts)
    merged_base, merged_lora = [], []
    for i in range(max_n):
        for s in range(total_splits):
            if i < parts[s]["n"]:
                merged_base.append(parts[s]["results_base"][i])
                merged_lora.append(parts[s]["results_lora"][i])
    n = len(merged_base)
    both_correct = sum(b and l for b, l in zip(merged_base, merged_lora))
    both_wrong   = sum(not b and not l for b, l in zip(merged_base, merged_lora))
    corrected    = sum(not b and l for b, l in zip(merged_base, merged_lora))
    misleading   = sum(b and not l for b, l in zip(merged_base, merged_lora))
    return {
        "n": n,
        "acc_base": sum(merged_base) / n * 100,
        "acc_lora": sum(merged_lora) / n * 100,
        "srp_rate": sum(p["srp_count"] for p in parts) / n * 100,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "corrected": corrected,
        "misleading": misleading,
    }


def _print_summary(m: dict, dataset: str, lora_dir: str, total_time: float = 0):
    log("\n" + "=" * 60)
    log(f"  数据集: {dataset}  |  样本数: {m['n']}" +
        (f"  |  耗时: {total_time:.0f}s" if total_time else ""))
    log(f"  LoRA:  {lora_dir}")
    log("=" * 60)
    log(f"  Base 准确率:   {m['acc_base']:5.1f}%  ({round(m['acc_base']*m['n']/100)}/{m['n']})")
    log(f"  LoRA 准确率:   {m['acc_lora']:5.1f}%  ({round(m['acc_lora']*m['n']/100)}/{m['n']})")
    log(f"  净增益:        {m['acc_lora'] - m['acc_base']:+.1f}%  ({m['corrected'] - m['misleading']:+d} 条)")
    log(f"  SRP 格式触发率: {m['srp_rate']:.0f}%")
    log()
    log(f"  四象限:")
    log(f"    both_correct : {m['both_correct']:3d}  ({m['both_correct']/m['n']*100:5.1f}%)")
    log(f"    corrected    : {m['corrected']:3d}  ({m['corrected']/m['n']*100:5.1f}%)  🟢")
    log(f"    both_wrong   : {m['both_wrong']:3d}  ({m['both_wrong']/m['n']*100:5.1f}%)")
    log(f"    misleading   : {m['misleading']:3d}  ({m['misleading']/m['n']*100:5.1f}%)  ⚠️")
    log("=" * 60)


# ── 单 GPU 评测核心 ───────────────────────────────────────────

def run_eval(args, split_id: int = 0, total_splits: int = 1, device: str = "cuda:0") -> dict:
    """在单张 GPU 上评测，返回结果 dict。多卡模式下由子进程调用。"""
    cfg = DATASET_CONFIGS[args.dataset]

    log(f"加载数据：{cfg['file']}（最多 {args.max_samples} 条）")
    data = load_data(cfg, args.max_samples, args.seed, args)
    if total_splits > 1:
        data = [d for i, d in enumerate(data) if i % total_splits == split_id]
        log(f"分片 {split_id}/{total_splits}，本片样本数：{len(data)}")
    else:
        log(f"实际样本数：{len(data)}")
    if not data:
        log("无数据，退出")
        return {}

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
        device_map=device, trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()
    log("模型加载完成\n")

    results_base, results_lora = [], []
    srp_count = 0
    t_start = time.time()

    for i, sample in enumerate(data):
        user = sample["user"]
        gold = sample["gold"]

        model.enable_adapter_layers()
        t0 = time.time()
        resp_lora = generate(model, tokenizer, user, args.max_new_tokens, device)
        t_lora = time.time() - t0

        model.disable_adapter_layers()
        t0 = time.time()
        resp_base = generate(model, tokenizer, user, args.max_new_tokens, device)
        t_base = time.time() - t0

        ok_lora = match_answer(resp_lora, gold, cfg["type"])
        ok_base = match_answer(resp_base, gold, cfg["type"])
        has_srp = "<SRP_START>" in resp_lora and "<SRP_END>" in resp_lora

        results_base.append(ok_base)
        results_lora.append(ok_lora)
        if has_srp:
            srp_count += 1

        n = i + 1
        acc_l = sum(results_lora) / n * 100
        acc_b = sum(results_base) / n * 100
        elapsed = time.time() - t_start
        eta = elapsed / n * (len(data) - n)

        mark_l = "✓" if ok_lora else "✗"
        mark_b = "✓" if ok_base else "✗"
        srp_flag = "SRP" if has_srp else "---"
        prefix = f"[GPU{split_id}]" if total_splits > 1 else ""

        log(f"\n{'─'*70}")
        log(
            f"{prefix}[{n:3d}/{len(data)}] base={mark_b} lora={mark_l} {srp_flag} "
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

    model.enable_adapter_layers()

    n = len(data)
    both_correct = sum(b and l for b, l in zip(results_base, results_lora))
    both_wrong   = sum(not b and not l for b, l in zip(results_base, results_lora))
    corrected    = sum(not b and l for b, l in zip(results_base, results_lora))
    misleading   = sum(b and not l for b, l in zip(results_base, results_lora))
    return {
        "split_id": split_id,
        "total_splits": total_splits,
        "dataset": args.dataset,
        "n": n,
        "acc_base": sum(results_base) / n * 100,
        "acc_lora": sum(results_lora) / n * 100,
        "srp_rate": srp_count / n * 100,
        "srp_count": srp_count,
        "results_base": results_base,
        "results_lora": results_lora,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "corrected": corrected,
        "misleading": misleading,
        "elapsed": time.time() - t_start,
    }


# ── 多卡并行调度 ──────────────────────────────────────────────

def _spawn_worker(gpu_id: int, split_id: int, total_splits: int, args, output_json: Path):
    """在指定 GPU 上启动子进程，返回 Popen。"""
    cmd = [
        sys.executable, "-u", __file__,
        "--dataset", args.dataset,
        "--lora_dir", args.lora_dir,
        "--base_model", args.base_model,
        "--max_samples", str(args.max_samples),
        "--max_new_tokens", str(args.max_new_tokens),
        "--seed", str(args.seed),
        "--_split_id", str(split_id),
        "--_total_splits", str(total_splits),
        "--_output_json", str(output_json),
    ]
    if args.math_subject:
        cmd.extend(["--math_subject", args.math_subject])
    if args.math_level:
        cmd.extend(["--math_level", args.math_level])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    return subprocess.Popen(cmd, env=env)


def run_multi_gpu(args, gpus: list):
    total_splits = len(gpus)
    log(f"多卡并行评测：{total_splits} 张 GPU {gpus}，共 {args.max_samples} 样本\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        procs, out_files = [], []
        for i, gpu_id in enumerate(gpus):
            out = tmpdir / f"split_{i}.json"
            procs.append((gpu_id, _spawn_worker(gpu_id, i, total_splits, args, out)))
            out_files.append(out)

        for gpu_id, proc in procs:
            ret = proc.wait()
            if ret != 0:
                log(f"GPU {gpu_id} 子进程异常退出，code={ret}")
                sys.exit(ret)

        parts = [json.loads(p.read_text()) for p in out_files]

    merged = _merge_parts(parts)
    _print_summary(merged, args.dataset, args.lora_dir)


# ── 入口 ────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 子进程模式（多卡时由 _spawn_worker 调起） ──
    if args._total_splits > 1 and args._output_json:
        result = run_eval(args, args._split_id, args._total_splits, device="cuda:0")
        Path(args._output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args._output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return

    # ── 多卡并行模式 ──
    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()] if args.gpus else []
    if len(gpus) > 1:
        run_multi_gpu(args, gpus)
        return

    # ── 单卡模式 ──
    device = f"cuda:{gpus[0]}" if gpus else "cuda:0"
    t_start = time.time()
    result = run_eval(args, device=device)
    if result:
        _print_summary(result, args.dataset, args.lora_dir, time.time() - t_start)


if __name__ == "__main__":
    main()
