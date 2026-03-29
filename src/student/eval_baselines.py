"""
SRP 对比 Baseline 评测脚本。

对同一数据集评测以下 5 种条件，帮助定位 SRP 的增益来源：

  B0  Base Qwen3-8B-Base（无任何修改）
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

    # 或直接多卡（与 scripts/run_baselines.sh 内部一致）：
    python -u src/student/eval_baselines.py \\
        --dataset gsm8k --max_samples 200 --gpus 0,1,2,3,4,5,6,7

    # 通用推理基准（需先有 data/raw/*.json，见 save_reasoning_benchmarks.py）：
    python -u src/student/eval_baselines.py \\
        --dataset mmlu_pro --max_samples 200 --gpus 0,1,2,3
"""

import argparse
import json
import os
import re
import random
import string
import subprocess
import sys
import tempfile
import time
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
    "musique": {
        "file": "data/raw/sample/musique_ans_train.jsonl",
        "type": "musique",
        # user/gold 由 load_data 内部从 question/answer/paragraphs 字段构造
    },
    "musique_full": {
        "file": "data/raw/musique_ans_train.jsonl",
        "type": "musique",
    },
    "super_gpqa": {
        "file": "data/raw/super_gpqa.json",
        "format": "json_splits",
        "json_splits": ["train"],
        "type": "mcq",
    },
    "mmlu_pro": {
        "file": "data/raw/mmlu_pro.json",
        "format": "json_splits",
        "json_splits": ["test"],
        "type": "mcq",
    },
    "bbeh": {
        "file": "data/raw/bbeh.json",
        "format": "json_splits",
        "json_splits": ["train"],
        "type": "bbeh",
    },
}


def log(msg: str = ""):
    print(msg, flush=True)


# ── 答案提取（与历史 eval 脚本同策略，便于论文口径一致）────────────

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


def extract_mcq_letter(text: str, num_choices: int) -> str:
    """多项选择题：选项字母为 A 起的连续 N 个（如 MMLU-Pro 10 选、SuperGPQA 可能多于 4 选）。"""
    text = re.split(r"<\|im_end\|>", text)[0].strip()
    valid = set(string.ascii_uppercase[: max(1, num_choices)])
    for m in reversed(list(re.finditer(
        r"(?:answer|choice|correct\s+option)[\s\w]*?[:\-]?\s*\(?([A-Z])\)?(?=\b|[\s\.\)\,])",
        text, re.I,
    ))):
        c = m.group(1).upper()
        if c in valid:
            return c
    m = re.search(r"\*\*([A-Z])\*\*|\(([A-Z])\)", text)
    if m:
        c = (m.group(1) or m.group(2)).upper()
        if c in valid:
            return c
    for c in reversed(re.findall(r"\b([A-Z])\b", text)):
        cu = c.upper()
        if cu in valid:
            return cu
    return ""


def match_bbeh_answer(pred: str, gold: str) -> bool:
    """BBEH：目标为短字符串 / 词；允许模型在 CoT 后给出答案，做规范化与边界匹配。"""
    text = re.split(r"<\|im_end\|>", pred)[0].strip()
    g = str(gold).strip()
    if not g:
        return False
    gn = " ".join(g.lower().split())
    pnorm = " ".join(text.lower().split())
    last = " ".join(text.strip().split("\n")[-1].lower().split()).rstrip(".")
    if last == gn or last.rstrip(".") == gn:
        return True
    if gn == pnorm:
        return True
    if len(gn) <= 3:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(gn)}(?![a-z0-9])", pnorm))
    if gn in pnorm:
        return True
    return False


def match_answer(pred: str, gold, dtype: str, mcq_n: int | None = None) -> bool:
    if dtype == "numeric":
        p = extract_gsm8k_answer(pred).replace(",", "")
        g = str(gold).strip().replace(",", "")
        if not p or not g: return False
        try: return float(p) == float(g)
        except ValueError: return p == g
    elif dtype == "choice":
        p = extract_choice(pred)
        m = re.match(r"([A-D])", str(gold).strip())
        return bool(p and m and p == m.group(1).upper())
    elif dtype == "musique":
        # gold 是 {"answer": str, "aliases": list}
        text = re.split(r"<\|im_end\|>", pred)[0].strip().lower()
        main_ans = str(gold.get("answer", "")).strip().lower()
        if main_ans and main_ans in text:
            return True
        for alias in gold.get("aliases", []):
            a = str(alias).strip().lower()
            if a and a in text:
                return True
        return False
    elif dtype == "mcq":
        n = mcq_n or 4
        p = extract_mcq_letter(pred, n)
        g = str(gold).strip().upper()[:1]
        return bool(p and g and p == g)
    elif dtype == "bbeh":
        return match_bbeh_answer(pred, str(gold))
    else:
        p = extract_hotpot_answer(pred).lower()
        g = str(gold).strip().lower()
        return g in p or p in g


# ── 数据加载 ─────────────────────────────────────────────────────

def _format_mcq_user(question: str, options: list) -> str:
    lines = [question.strip(), "", "Options:"]
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        o = opt.strip() if isinstance(opt, str) else str(opt).strip()
        lines.append(f"{letter}. {o}")
    lines.append("")
    lines.append("Respond with only the letter of the correct option (e.g. A).")
    return "\n".join(lines)


def _mcq_letter_from_obj(obj: dict, n_opts: int) -> str:
    if obj.get("answer_letter"):
        return str(obj["answer_letter"]).strip().upper()[:1]
    ans = obj.get("answer")
    if ans is not None:
        s = str(ans).strip().upper()
        if len(s) == 1 and s in string.ascii_uppercase[:n_opts]:
            return s
    idx = obj.get("answer_index")
    if idx is not None:
        try:
            i = int(idx)
        except (TypeError, ValueError):
            i = -1
        if 0 <= i < n_opts:
            return string.ascii_uppercase[i]
    return ""


def _format_musique_user(obj: dict) -> str:
    """
    MuSiQue user 消息格式：把 is_supporting=True 的段落作为上下文，
    然后接多跳问题。
    """
    paras = obj.get("paragraphs", [])
    supporting = [p for p in paras if p.get("is_supporting")]
    # 去重（按 title）
    seen = set()
    ctx_parts = []
    for p in supporting:
        title = p.get("title", "")
        text = p.get("paragraph_text", "")
        key = title.strip()
        if key not in seen and text.strip():
            seen.add(key)
            ctx_parts.append(f"**{title}**\n{text.strip()}")
    ctx = "\n\n".join(ctx_parts)
    question = obj.get("question", "").strip()
    if ctx:
        return f"Use the following passages to answer the question.\n\n{ctx}\n\nQuestion: {question}"
    return question


def load_data(cfg: dict, max_samples: int, seed: int) -> list:
    data = []
    is_musique = cfg.get("type") == "musique"
    fmt = cfg.get("format", "jsonl")

    if fmt == "json_splits":
        path = Path(cfg["file"])
        if not path.is_file():
            raise FileNotFoundError(f"数据文件不存在: {path.resolve()}")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        raw_rows = []
        for sp in cfg.get("json_splits", ["train"]):
            raw_rows.extend(bundle.get(sp, []))
        for obj in raw_rows:
            if cfg["type"] == "mcq":
                q = (obj.get("question") or "").strip()
                opts = obj.get("options") or []
                if isinstance(opts, str):
                    opts = [x.strip() for x in re.split(r"[\n\|;,]", opts) if x.strip()]
                if not q or not opts:
                    continue
                letter = _mcq_letter_from_obj(obj, len(opts))
                if not letter:
                    continue
                user = _format_mcq_user(q, opts)
                data.append({
                    "user": user, "gold": letter, "sr_prompt": "",
                    "mcq_n": len(opts),
                })
            elif cfg["type"] == "bbeh":
                inp = (obj.get("input") or "").strip()
                target = (obj.get("target") or "").strip()
                if not inp or not target:
                    continue
                data.append({"user": inp, "gold": target, "sr_prompt": ""})
        random.seed(seed)
        random.shuffle(data)
        return data[:max_samples]

    with open(cfg["file"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)

            if is_musique:
                if not obj.get("answerable", True):
                    continue
                user = _format_musique_user(obj)
                gold = {
                    "answer": obj.get("answer", ""),
                    "aliases": obj.get("answer_aliases", []),
                }
                sr_prompt = ""
            else:
                user = (obj.get(cfg.get("user_key", "user")) or "").strip()
                gold = (obj.get(cfg.get("gold_key", "answer")) or "").strip()
                sr_prompt = (obj.get(cfg.get("srp_key", "sr_prompt")) or "").strip()

            if user and gold:
                rec = {"user": user, "gold": gold, "sr_prompt": sr_prompt}
                data.append(rec)

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
        ok = match_answer(resp, gold, dtype, sample.get("mcq_n"))
        corrects.append(ok)

        # 实时进度
        n = i + 1
        acc = sum(corrects) / n * 100
        log(f"  [{mode}] [{n:3d}/{len(data)}] {'✓' if ok else '✗'}  acc={acc:5.1f}%")

    model.enable_adapter_layers()  # 恢复默认状态
    return {"mode": mode, "n": len(data), "acc": sum(corrects) / len(data) * 100,
            "corrects": corrects}


def load_tokenizer_and_peft(args, device: str):
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
    return tokenizer, model


def filter_b2_if_no_srp(modes: list, data: list, log_coverage: bool = True) -> list:
    if "B2" not in modes:
        return modes
    n = len(data)
    has_srp = sum(1 for d in data if d.get("sr_prompt"))
    if log_coverage and n:
        log(f"Oracle SRP 覆盖率：{has_srp}/{n} ({has_srp/n*100:.0f}%)")
    if has_srp == 0:
        if log_coverage:
            log("  ⚠️  数据中无 sr_prompt，跳过 B2")
        return [m for m in modes if m != "B2"]
    return modes


def merge_shard_parts(parts: list, modes_order: list, n: int, total_splits: int) -> list:
    parts = sorted(parts, key=lambda x: x["split_id"])
    merged_results = []
    for mode in modes_order:
        buf = [None] * n
        for part in parts:
            sid = part["split_id"]
            for rec in part["per_mode"]:
                if rec["mode"] != mode:
                    continue
                for j, ok in enumerate(rec["corrects"]):
                    g = j * total_splits + sid
                    if g < n:
                        buf[g] = ok
                break
        if any(x is None for x in buf):
            raise RuntimeError(f"合并分片失败：模式 {mode} 存在空缺")
        acc = sum(buf) / n * 100
        merged_results.append({"mode": mode, "n": n, "acc": acc, "corrects": buf})
    return merged_results


def print_summary(all_results: list, n: int, args, total_time: float) -> None:
    log(f"\n{'='*65}")
    log(f"  数据集: {args.dataset}  |  样本数: {n}  |  总耗时: {total_time:.0f}s")
    log(f"  LoRA:  {args.lora_dir}")
    log(f"{'='*65}")
    log(f"  {'模式':<6}  {'准确率':>8}  {'正确':>6}  {'描述'}")
    log(f"  {'─'*60}")
    mode_desc_short = {
        "B0": "Base（无修改）",
        "B1": "Base + CoT prompt（zero-shot）",
        "B2": "Base + Oracle SRP（teacher hint）",
        "B3": "Base（disable_adapter，同 B0）",
        "B4": "SRP-LoRA（our method）",
    }
    for r in all_results:
        log(f"  {r['mode']:<6}  {r['acc']:>7.1f}%  {round(r['acc']*n/100):>5}/{n}  "
            f"{mode_desc_short.get(r['mode'], r['mode'])}")
    b4 = next((r for r in all_results if r["mode"] == "B4"), None)
    if b4:
        log(f"\n  SRP-LoRA vs 各 Baseline：")
        for r in all_results:
            if r["mode"] == "B4":
                continue
            delta = b4["acc"] - r["acc"]
            sign = "🟢" if delta > 0 else ("🔴" if delta < 0 else "➡️")
            log(f"    vs {r['mode']}:  {delta:+.1f}%  {sign}")
    log(f"{'='*65}")


def _shard_worker_cmd(args: argparse.Namespace, split_id: int, total_splits: int,
                      output_json: Path, modes_csv: str) -> list:
    return [
        sys.executable, "-u", __file__,
        "--dataset", args.dataset,
        "--base_model", args.base_model,
        "--lora_dir", args.lora_dir,
        "--max_samples", str(args.max_samples),
        "--max_new_tokens", str(args.max_new_tokens),
        "--seed", str(args.seed),
        "--modes", modes_csv,
        "--_split_id", str(split_id),
        "--_total_splits", str(total_splits),
        "--_output_json", str(output_json),
    ]


def run_shard_worker(args: argparse.Namespace) -> None:
    cfg = DATASET_CONFIGS[args.dataset]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    data_full = load_data(cfg, args.max_samples, args.seed)
    n_full = len(data_full)
    modes = filter_b2_if_no_srp(modes, data_full, log_coverage=False)

    S = args._total_splits
    sid = args._split_id
    data = [d for i, d in enumerate(data_full) if i % S == sid]

    log(f"[分片 {sid}/{S}] 样本 {len(data)}/{n_full} | {args.dataset}")
    tokenizer, model = load_tokenizer_and_peft(args, "cuda:0")
    per_mode = []
    for mode in modes:
        r = eval_one_mode(mode, data, model, tokenizer,
                          cfg["type"], args.max_new_tokens, "cuda:0")
        per_mode.append({"mode": mode, "corrects": r["corrects"]})

    out = {"split_id": sid, "total_splits": S, "n_full": n_full, "per_mode": per_mode}
    Path(args._output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args._output_json).write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8"
    )


def run_multi_gpu(args: argparse.Namespace, gpus: list) -> None:
    cfg = DATASET_CONFIGS[args.dataset]
    log(f"加载数据：{cfg['file']}（最多 {args.max_samples} 条）")
    data_full = load_data(cfg, args.max_samples, args.seed)
    n = len(data_full)
    log(f"实际样本数：{n}\n")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    modes = filter_b2_if_no_srp(modes, data_full)
    modes_csv = ",".join(modes)

    S = len(gpus)
    log(f"\n多卡评测：{S} 路子进程，GPUs {gpus}\n")

    t_start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        procs = []
        out_files = []
        for i, gpu_id in enumerate(gpus):
            out_f = tmpdir / f"shard_{i}.json"
            cmd = _shard_worker_cmd(args, i, S, out_f, modes_csv)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            procs.append(subprocess.Popen(cmd, env=env))
            out_files.append(out_f)
        for i, proc in enumerate(procs):
            ret = proc.wait()
            if ret != 0:
                raise SystemExit(f"baseline 分片子进程失败 GPU={gpus[i]} code={ret}")
        parts = [json.loads(p.read_text(encoding="utf-8")) for p in out_files]

    merged = merge_shard_parts(parts, modes, n, S)
    print_summary(merged, n, args, time.time() - t_start)


# ── main ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SRP baseline comparison")
    p.add_argument("--base_model", default="model/Qwen3-8B-Base")
    p.add_argument("--lora_dir", default="outputs/qwen3_sr_lora_v3_base")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="musique")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modes", type=str, default="B0,B1,B2,B4",
                   help="要评测的模式，逗号分隔。可选：B0,B1,B2,B3,B4")
    p.add_argument("--gpus", type=str, default="",
                   help="逗号分隔 GPU id；多张卡时按样本 index 分片并行（子进程内 cuda:0）")
    p.add_argument("--_split_id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--_total_splits", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--_output_json", type=str, default="", help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()

    if args._output_json:
        run_shard_worker(args)
        return

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if len(gpus) > 1:
        run_multi_gpu(args, gpus)
        return

    cfg = DATASET_CONFIGS[args.dataset]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    log(f"加载数据：{cfg['file']}（最多 {args.max_samples} 条）")
    data = load_data(cfg, args.max_samples, args.seed)
    log(f"实际样本数：{len(data)}\n")

    modes = filter_b2_if_no_srp(modes, data)

    device = f"cuda:{gpus[0]}" if gpus else args.device
    log(f"\n加载模型：{args.base_model} + {args.lora_dir}")
    tokenizer, model = load_tokenizer_and_peft(args, device)
    log("模型加载完成\n")

    all_results = []
    t_start = time.time()

    for mode in modes:
        mode_desc = {
            "B0": "Base Qwen3-8B-Base（无修改）",
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
                               cfg["type"], args.max_new_tokens, device)
        result["time"] = time.time() - t0
        all_results.append(result)

    print_summary(all_results, len(data), args, time.time() - t_start)


if __name__ == "__main__":
    main()
