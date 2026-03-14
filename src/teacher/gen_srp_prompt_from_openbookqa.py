"""
为 OpenBookQA 数据集生成 Self-RePrompt 三元组（user / sr_prompt / answer）。

数据特征
--------
- 问题形式：多项选择（A/B/C/D），考察基础科学常识
- 核心字段：fact1（回答问题所需的科学事实），仅供 teacher 参考，
  不得直接写入 sr_prompt（避免泄题）
- sr_prompt 应给出推理方向/知识域提示，而非具体事实或答案

输入
----
    data/raw/openbookqa_additional.json
    格式: {
        "train": [{"id","question_stem","choices":{"text":[],"label":[]},"answerKey","fact1",...}, ...]
        "validation": [...], "test": [...]
    }

输出
----
    data/srp_prompt/openbookqa_train_reprompt.jsonl
    每行: {"user": <格式化问题+选项>, "sr_prompt": <推理策略>, "answer": <答案字母+文本>}

典型用法
--------
    # 全量生成
    python src/teacher/gen_srp_prompt_from_openbookqa.py

    # 只处理 train 的前 200 条（快速验证）
    python src/teacher/gen_srp_prompt_from_openbookqa.py --max_samples 200

    # 使用 Qwen3 模型
    python src/teacher/gen_srp_prompt_from_openbookqa.py --model Qwen/Qwen3-235B-A22B
"""

import argparse
import json
import os
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
        description="为 OpenBookQA 生成 Self-RePrompt 三元组（srp_prompt）。"
    )
    parser.add_argument("--input",  type=str,
                        default="data/raw/openbookqa_additional.json")
    parser.add_argument("--split",  type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="使用哪个 split（默认 train）。")
    parser.add_argument("--output", type=str,
                        default="data/srp_prompt/openbookqa_train_reprompt.jsonl")
    parser.add_argument("--model",  type=str, default="deepseek-chat",
                        help="Teacher LLM 模型名。")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最多处理多少条样本（用于快速验证）。")
    parser.add_argument("--workers", type=int, default=8,
                        help="并行线程数。")
    parser.add_argument("--min_clarity", type=float, default=0.8,
                        help="仅保留 clarity >= 此值的样本（过滤低质量题目）。")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# 数据加载 & 格式化
# ─────────────────────────────────────────────────────────────
def format_user(item: Dict) -> str:
    """将 question_stem + choices 格式化为自然问题文本。"""
    stem = item["question_stem"].strip().rstrip("?")
    lines = [f"{stem}?"] if not stem.endswith("?") else [stem]
    for label, text in zip(item["choices"]["label"], item["choices"]["text"]):
        lines.append(f"  {label}) {text}")
    return "\n".join(lines)


def format_answer(item: Dict) -> str:
    """返回 'D) plants sprouting, blooming and wilting' 形式的标准答案。"""
    key = item["answerKey"]
    idx = item["choices"]["label"].index(key)
    text = item["choices"]["text"][idx]
    return f"{key}) {text}"


def load_data(path: str, split: str,
              max_samples: Optional[int],
              min_clarity: float) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw[split]
    # 过滤低质量题目
    items = [it for it in items if it.get("clarity", 0) >= min_clarity]
    if max_samples:
        items = items[:max_samples]
    return items


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
# Teacher prompt 设计
# ─────────────────────────────────────────────────────────────
def build_teacher_messages(user_text: str, fact1: str) -> List[Dict[str, str]]:
    """
    Teacher 知道 fact1（核心科学事实），但生成的 sr_prompt 不得直接引用该事实，
    只能给出指向正确知识域的推理方向提示。
    """
    system_prompt = (
        "You are an expert prompt engineer for science multiple-choice questions.\n\n"
        "You will be given:\n"
        "  1. A question with four choices (A/B/C/D)\n"
        "  2. A PRIVATE core fact that underlies the correct answer "
        "(do NOT reveal this fact in your output)\n\n"
        "Your task: write a SHORT guiding instruction (1–2 sentences, ≤ 45 words) "
        "that helps another LLM pick the right choice — WITHOUT giving away the answer.\n\n"
        "Rules:\n"
        "- Point to the relevant science domain or reasoning process (cause-effect, "
        "characteristic, cycle, classification, etc.)\n"
        "- Do NOT repeat or paraphrase the core fact directly\n"
        "- Do NOT name the correct choice or its text\n"
        "- Do NOT restate the question\n"
        "- English only; output ONLY the instruction text\n\n"
        "Strategy types (pick the most relevant):\n"
        "- Causal reasoning: identify what causes or enables the described outcome\n"
        "- Characteristic lookup: recall a defining property of the subject\n"
        "- Process/cycle: trace the sequence of a natural process step by step\n"
        "- Classification: determine which category or domain the subject belongs to\n"
        "- Elimination: rule out choices that contradict known scientific principles\n\n"
        "Examples of good instructions:\n\n"
        "Q: Which property of sound changes when you move closer to the source?\n"
        "Instruction: Think about how the intensity or volume of a wave changes "
        "with distance from its source.\n\n"
        "Q: The sun is responsible for [choices about plants/animals/etc.]\n"
        "Core fact (private): the sun is the source of energy for physical cycles on Earth\n"
        "Instruction: Identify which answer describes an energy-driven natural cycle "
        "on Earth that depends on an external energy source.\n\n"
        "Q: When food is reduced in the stomach [choices about digestion/reading]\n"
        "Core fact (private): digestion is when stomach acid breaks down food\n"
        "Instruction: Focus on the biological process that physically or chemically "
        "breaks down food in the digestive system."
    )

    user_prompt = (
        f"Question:\n{user_text}\n\n"
        f"Core fact (private, do NOT reveal): {fact1}\n\n"
        "Write the guiding instruction (as described above). Output ONLY the instruction text."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ─────────────────────────────────────────────────────────────
# API 调用
# ─────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_llm(client: OpenAI, model: str,
             messages: List[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=120,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────
# 单条处理
# ─────────────────────────────────────────────────────────────
def process_one(
    client: OpenAI, model: str, idx: int, item: Dict
) -> Tuple[int, Dict]:
    user_text = format_user(item)
    answer    = format_answer(item)
    fact1     = item.get("fact1", "").strip()

    messages  = build_teacher_messages(user_text, fact1)
    sr_prompt = call_llm(client, model, messages)

    record = {
        "id":        item["id"],
        "user":      user_text,
        "sr_prompt": sr_prompt,
        "answer":    answer,
        "fact1":     fact1,        # 保留供后续分析，训练时忽略
        "clarity":   item.get("clarity", 0),
    }
    return idx, record


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()
    client = get_client()

    print(f"加载数据：{args.input}  split={args.split}  min_clarity={args.min_clarity}")
    items = load_data(args.input, args.split, args.max_samples, args.min_clarity)
    total = len(items)
    print(f"有效样本：{total}，并行 workers={args.workers}")
    if total == 0:
        print("无有效样本，退出。")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_indices = load_done_indices(out_path)
    if done_indices:
        print(f"已完成 {len(done_indices)} 条，断点续传，跳过已有结果。")

    todo = [(idx, item) for idx, item in enumerate(items, start=1)
            if idx not in done_indices]
    print(f"待生成：{len(todo)} 条")
    if not todo:
        print("全部已完成。")
        return

    write_lock  = threading.Lock()
    done_count  = len(done_indices)

    def write_record(rec: Dict) -> None:
        nonlocal done_count
        with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
            done_count += 1
            if done_count % 100 == 0 or done_count == total:
                print(f"进度 {done_count}/{total} ({done_count/total*100:.1f}%) ...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, client, args.model, idx, item): idx
            for idx, item in todo
        }
        for future in as_completed(futures):
            try:
                _, record = future.result()
                write_record(record)
            except Exception as e:
                print(f"[ERROR] 样本处理失败：{e}")

    print(f"完成。{total} 条已写入 {out_path}")


if __name__ == "__main__":
    main()
