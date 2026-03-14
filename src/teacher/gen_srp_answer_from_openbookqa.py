"""
基于 srp_prompt 为 OpenBookQA 生成 teacher 回答（srp_answer / direct_answer），
并打上四象限标签，便于后续筛选训练数据。

输入
----
    data/srp_prompt/openbookqa_train_reprompt.jsonl
    每行: {"id","user","sr_prompt","answer","fact1","clarity"}

输出
----
    data/srp_prompt_with_answer/openbookqa_train_with_srp_answer.jsonl
    每行新增字段:
        "direct_answer" : teacher 仅用 user 的回答
        "srp_answer"    : teacher 用 user + sr_prompt 的回答
        "ok_direct"     : direct 是否选对（bool）
        "ok_srp"        : srp 是否选对（bool）
        "quadrant"      : both_correct / both_wrong / misleading / corrected

四象限定义
----------
    both_correct : direct ✓  srp ✓
    both_wrong   : direct ✗  srp ✗
    misleading   : direct ✓  srp ✗   （sr_prompt 带偏了）
    corrected    : direct ✗  srp ✓   （sr_prompt 纠正了错误）

典型用法
--------
    python src/teacher/gen_srp_answer_from_openbookqa.py

    # 只跑前 100 条验证
    python src/teacher/gen_srp_answer_from_openbookqa.py --max_samples 100
"""

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 OpenBookQA 生成 srp_answer 并打四象限标签。"
    )
    parser.add_argument("--data_file", type=str,
                        default="data/srp_prompt/openbookqa_train_reprompt.jsonl")
    parser.add_argument("--output_file", type=str,
                        default="data/srp_prompt_with_answer/openbookqa_train_with_srp_answer.jsonl")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="多选题推理不需要太长，256 足够。")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# 答案提取 & 评测
# ─────────────────────────────────────────────────────────────
_ANSWER_RE = re.compile(
    r"""
    (?:
        (?:answer|choice|option|select(?:ed)?|correct)[\s\w]*?[:\-]?\s*  # 可选前缀
        |^|\b                                                             # 行首或词边界
    )
    \(?([A-D])\)?        # 捕获大写 A/B/C/D（不加 IGNORECASE，避免匹配 correct→c 等词内字母）
    (?![A-Za-z])         # 负向前瞻：字母后不能紧跟其他字母（排除 After→A、best→B 等误判）
    """,
    re.VERBOSE,          # 注意：不使用 IGNORECASE
)


def extract_choice(text: str) -> str:
    """
    从模型输出中提取最终选项字母 (A/B/C/D)。
    策略（优先级从高到低）：
      1. 最后一个显式 "answer is X" / "The answer: X" 形式
      2. 最后一个带括号 (A) 或加粗 **A** 的字母
      3. 整段文本最后一个独立字母 A-D
    """
    text = text.strip()
    if not text:
        return ""

    # 策略 1：寻找 answer/choice/correct + 字母
    for m in reversed(list(_ANSWER_RE.finditer(text))):
        return m.group(1).upper()

    # 策略 2：**A** 或 (A)
    m = re.search(r"\*\*([A-D])\*\*|\(([A-D])\)", text, re.IGNORECASE)
    if m:
        return (m.group(1) or m.group(2)).upper()

    # 策略 3：最后一个孤立的 A-D
    matches = re.findall(r"\b([A-D])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    return ""


def gold_letter(answer_str: str) -> str:
    """从 'D) plants sprouting...' 提取字母 'D'。"""
    m = re.match(r"([A-D])", answer_str.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else ""


def answer_match(pred_text: str, gold_str: str) -> bool:
    pred = extract_choice(pred_text)
    gold = gold_letter(gold_str)
    return bool(pred and gold and pred == gold)


def classify_quadrant(ok_direct: bool, ok_srp: bool) -> str:
    if ok_direct and ok_srp:     return "both_correct"
    if not ok_direct and ok_srp: return "corrected"
    if ok_direct and not ok_srp: return "misleading"
    return "both_wrong"


# ─────────────────────────────────────────────────────────────
# 数据加载 & 断点续传
# ─────────────────────────────────────────────────────────────
def load_data(path: str, max_samples: Optional[int]) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("user") or not obj.get("sr_prompt") or not obj.get("answer"):
                continue
            data.append(obj)
            if max_samples and len(data) >= max_samples:
                break
    return data


def load_done_indices(out_path: Path) -> set:
    done: set = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if line.strip():
                done.add(i)
    return done


# ─────────────────────────────────────────────────────────────
# API 调用
# ─────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def generate(client: OpenAI, model: str,
             messages: List[Dict], max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def build_direct_messages(user: str) -> List[Dict]:
    system = (
        "You are a helpful assistant. Answer the following multiple-choice question. "
        "Think step by step, then clearly state your final answer as: "
        "\"The answer is X\" (where X is A, B, C, or D)."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def build_srp_messages(user: str, sr_prompt: str) -> List[Dict]:
    system = (
        "You are a helpful assistant. Answer the following multiple-choice question. "
        "Think step by step, then clearly state your final answer as: "
        "\"The answer is X\" (where X is A, B, C, or D)."
    )
    combined = f"{user}\n\nHint: {sr_prompt}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": combined}]


# ─────────────────────────────────────────────────────────────
# 单条处理
# ─────────────────────────────────────────────────────────────
def process_one(
    client: OpenAI, model: str, max_tokens: int,
    idx: int, sample: Dict
) -> Tuple[int, Dict]:
    user      = sample["user"]
    sr_prompt = sample["sr_prompt"]
    gold      = sample["answer"]

    direct_answer = generate(client, model, build_direct_messages(user), max_tokens)
    srp_answer    = generate(client, model, build_srp_messages(user, sr_prompt), max_tokens)

    ok_direct = answer_match(direct_answer, gold)
    ok_srp    = answer_match(srp_answer,    gold)

    record = {
        **sample,
        "direct_answer": direct_answer,
        "srp_answer":    srp_answer,
        "ok_direct":     ok_direct,
        "ok_srp":        ok_srp,
        "quadrant":      classify_quadrant(ok_direct, ok_srp),
    }
    return idx, record


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()
    client = get_client()

    print(f"加载数据：{args.data_file}")
    data  = load_data(args.data_file, args.max_samples)
    total = len(data)
    if total == 0:
        print("无有效样本，退出。")
        return

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_indices = load_done_indices(out_path)
    if done_indices:
        print(f"已完成 {len(done_indices)} 条，断点续传。")

    todo = [(idx, s) for idx, s in enumerate(data, start=1)
            if idx not in done_indices]
    print(f"待生成：{len(todo)}/{total}，workers={args.workers}")
    if not todo:
        print("全部已完成。")
        return

    write_lock = threading.Lock()
    done_count = len(done_indices)
    stats = {"both_correct": 0, "both_wrong": 0, "misleading": 0, "corrected": 0}

    def write_record(rec: Dict) -> None:
        nonlocal done_count
        with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
            done_count += 1
            stats[rec["quadrant"]] += 1
            if done_count % 100 == 0 or done_count == total:
                pct  = done_count / total * 100
                corr = stats["both_correct"] + stats["corrected"]
                n    = sum(stats.values())
                print(
                    f"进度 {done_count}/{total} ({pct:.1f}%)  |  "
                    f"srp 准确: {corr}/{n} ({corr/n*100:.1f}%)  "
                    f"corrected={stats['corrected']}  misleading={stats['misleading']}"
                )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, client, args.model,
                            args.max_new_tokens, idx, sample): idx
            for idx, sample in todo
        }
        for future in as_completed(futures):
            try:
                _, record = future.result()
                write_record(record)
            except Exception as e:
                print(f"[ERROR] 样本处理失败：{e}")

    print(f"\n完成。{total} 条写入 {out_path}")
    total_n = sum(stats.values())
    if total_n:
        print(f"四象限统计（已完成 {total_n} 条）：")
        for q, n in stats.items():
            print(f"  {q:15s}: {n:4d}  ({n/total_n*100:.1f}%)")


if __name__ == "__main__":
    main()
