"""
在已有 user / sr_prompt / answer(gold) 数据上，调用硅基流动 API（如 DeepSeek），
根据 sr_prompt 指令对 user 问题生成模型回复，得到 sr_answer。
输出四字段 JSONL：user, sr_prompt, sr_answer, answer（gold）。
用于构造「用户输入 → sr_prompt → 基于 sr_prompt 的回复」的完整训练/评测数据。
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 sr_prompt 调用 API 生成 sr_answer，输出 user / sr_prompt / sr_answer / answer 四字段 JSONL。"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSONL，每行 {user, sr_prompt, answer}，answer 为 gold。",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSONL，每行 {user, sr_prompt, sr_answer, answer}。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="DeepSeek 官方模型名称（chat/completion 模型）。",
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
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY（DeepSeek 官方 API 密钥）")
    base_url = "https://api.deepseek.com"
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


def build_messages(user: str, sr_prompt: str) -> List[Dict[str, str]]:
    """sr_prompt 作为 system，user 作为 user 消息。"""
    return [
        {"role": "system", "content": sr_prompt},
        {"role": "user", "content": user},
    ]


def get_sr_reply(
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
) -> Tuple[int, Dict[str, str]]:
    """返回 (idx, record)，record 含 user, sr_prompt, sr_answer, answer。"""
    user = sample["user"]
    sr_prompt = sample["sr_prompt"]
    gold = sample["answer"]
    sr_answer = get_sr_reply(client, model_name, user, sr_prompt, max_new_tokens)
    record = {
        "user": user,
        "sr_prompt": sr_prompt,
        "sr_answer": sr_answer,
        "answer": gold,
    }
    return idx, record


def main() -> None:
    args = parse_args()
    client = get_client()

    print(f"加载数据：{args.input}")
    data = load_data(args.input, args.max_samples)
    total = len(data)
    print(f"待处理：{total} 条，workers={args.workers}，输出四字段 -> {args.output}")
    if total == 0:
        print("没有有效样本，退出。")
        return

    results: List[Tuple[int, Dict[str, str]]] = []
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
    records = [r[1] for r in results]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"完成。共 {total} 条，已写入 {args.output}（字段：user, sr_prompt, sr_answer, answer）")


if __name__ == "__main__":
    main()
