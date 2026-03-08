"""
根据已有 sr_prompt 数据，调用硅基流动 API（DeepSeek）按 sr_prompt 指令回答问题，
仅保留模型回答正确的条目，输出为 JSONL（user, sr_prompt, answer）。
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 API 按 sr_prompt 生成回答，仅保留回答正确的条目。"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSONL，每行 {user, sr_prompt, answer}。",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSONL，仅包含模型答对的条目，格式同输入。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V3.2",
        help="硅基流动上的模型名，默认 DeepSeek。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多处理条数，默认全部。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="单条生成的最大 token 数。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行请求线程数。",
    )
    return parser.parse_args()


def get_client() -> OpenAI:
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("请设置环境变量 SILICONFLOW_API_KEY")
    base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def load_data(path: str, max_samples: Optional[int]) -> List[Dict[str, str]]:
    data: List[Dict[str, str]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            user = (obj.get("user") or "").strip()
            sr_prompt = (obj.get("sr_prompt") or "").strip()
            answer = (obj.get("answer") or "").strip()
            if not user or not answer or not sr_prompt:
                continue
            data.append({"user": user, "sr_prompt": sr_prompt, "answer": answer})
            if max_samples is not None and len(data) >= max_samples:
                break
    return data


def normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r'^[\s\"\'\(\)\[\]]+', "", t)
    t = re.sub(r'[\s\"\'\(\)\[\]\.\,\;\:\!\?]+$', "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def answer_match(pred: str, gold: str) -> bool:
    p = normalize(pred)
    g = normalize(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if g in p:
        return True
    return False


def build_messages(user: str, sr_prompt: str) -> List[Dict[str, str]]:
    """sr_prompt 作为 system，user 作为 user 消息（与 reson 评测一致）。"""
    return [
        {"role": "system", "content": sr_prompt},
        {"role": "user", "content": user},
    ]


def get_reply(
    client: OpenAI,
    model_name: str,
    user: str,
    sr_prompt: str,
    max_new_tokens: int,
) -> str:
    messages = build_messages(user, sr_prompt)
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    return (content or "").strip()


def process_one(
    client: OpenAI,
    model_name: str,
    max_new_tokens: int,
    idx: int,
    sample: Dict[str, str],
) -> Tuple[int, bool, Dict[str, str]]:
    """返回 (idx, is_correct, sample)。"""
    user = sample["user"]
    sr_prompt = sample["sr_prompt"]
    gold = sample["answer"]
    pred = get_reply(client, model_name, user, sr_prompt, max_new_tokens)
    ok = answer_match(pred, gold)
    return idx, ok, sample


def main() -> None:
    args = parse_args()
    client = get_client()

    print(f"加载数据：{args.input}")
    data = load_data(args.input, args.max_samples)
    total = len(data)
    print(f"待处理：{total} 条，workers={args.workers}，仅保留答对的条目 -> {args.output}")
    if total == 0:
        print("没有有效样本，退出。")
        return

    results: List[Tuple[int, bool, Dict[str, str]]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one,
                client,
                args.model,
                args.max_new_tokens,
                idx,
                sample,
            ): idx
            for idx, sample in enumerate(data, start=1)
        }
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 50 == 0 or done == total:
                print(f"已处理 {done}/{total} 条...")

    results.sort(key=lambda x: x[0])
    correct_records = [r[2] for r in results if r[1]]
    n_correct = len(correct_records)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in correct_records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"完成。总条数: {total}，答对: {n_correct}，已写入 {args.output}")


if __name__ == "__main__":
    main()
