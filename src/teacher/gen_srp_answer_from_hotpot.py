"""
使用 DeepSeek/其他 teacher LLM 在 Hotpot QA 上生成 srp_answer（基于 sr_prompt 的 teacher 回答），
并为每条样本打上 direct/sr 准确性以及四象限标签，便于后续筛选训练数据。

输入：
    data/srp_prompt/hotpot_train_qa_2000_reprompt.jsonl
    每行: {"user": str, "sr_prompt": str, "answer": str}

输出：
    默认输出到：
        data/srp_prompt_with_answer/hotpot_train_qa_2000_with_srp_answer.jsonl

    JSONL，每行包含：
        {
          "user": ...,
          "sr_prompt": ...,
          "answer": ...,
          "direct_answer": str,    # teacher 仅用 user 的回答
          "srp_answer": str,       # teacher 用 user+sr_prompt 的回答
          "ok_direct": bool,       # direct 是否答对
          "ok_srp": bool,          # srp_answer 是否答对
          "quadrant": str          # both_correct / both_wrong / misleading / corrected
        }
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
        description="在 Hotpot QA 上生成 srp_answer，并打上 direct/srp 四象限标签。"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="用于生成回答的 teacher 模型名称（DeepSeek 官方模型名）。",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/srp_prompt/hotpot_train_qa_2000_reprompt.jsonl",
        help="包含 {user, sr_prompt, answer} 的 JSONL 文件路径。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多处理多少条样本。默认不限制。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=350,
        help="每条样本生成的最大 new tokens。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行请求的线程数。",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/srp_prompt_with_answer/hotpot_train_qa_2000_with_srp_answer.jsonl",
        help="输出 JSONL 路径。",
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


def normalize(text: str) -> str:
    import re

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


def generate_answer_api(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    return (content or "").strip()


def build_messages_direct(user: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": user}]


def build_messages_with_srp(user: str, sr_prompt: str) -> List[Dict[str, str]]:
    prompt = f"{user}\n\n{sr_prompt}"
    return [{"role": "user", "content": prompt}]


def classify_quadrant(ok_direct: bool, ok_srp: bool) -> str:
    if ok_direct and ok_srp:
        return "both_correct"
    if not ok_direct and not ok_srp:
        return "both_wrong"
    if ok_direct and not ok_srp:
        return "misleading"
    return "corrected"


def process_one(
    client: OpenAI,
    model_name: str,
    max_new_tokens: int,
    idx: int,
    sample: Dict[str, str],
) -> Tuple[int, Dict[str, object]]:
    user = sample["user"]
    sr_prompt = sample["sr_prompt"]
    gold = sample["answer"]

    msg_direct = build_messages_direct(user)
    direct_answer = generate_answer_api(client, model_name, msg_direct, max_new_tokens)

    msg_srp = build_messages_with_srp(user, sr_prompt)
    srp_answer = generate_answer_api(client, model_name, msg_srp, max_new_tokens)

    ok_direct = answer_match(direct_answer, gold)
    ok_srp = answer_match(srp_answer, gold)
    quadrant = classify_quadrant(ok_direct, ok_srp)

    record: Dict[str, object] = {
        "user": user,
        "sr_prompt": sr_prompt,
        "answer": gold,
        "direct_answer": direct_answer,
        "srp_answer": srp_answer,
        "ok_direct": ok_direct,
        "ok_srp": ok_srp,
        "quadrant": quadrant,
    }
    return idx, record


def main() -> None:
    args = parse_args()
    client = get_client()

    print(f"加载数据：{args.data_file}" + (f"（最多 {args.max_samples} 条）" if args.max_samples else "（全部）"))
    data = load_data(args.data_file, args.max_samples)
    total = len(data)
    print(f"实际生成样本数：{total}，并行 workers={args.workers}")
    if total == 0:
        print("没有有效样本，退出。")
        return

    results: List[Tuple[int, Dict[str, object]]] = []
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
            if done % 10 == 0 or done == total:
                print(f"已生成 {done}/{total} 条...")

    results.sort(key=lambda x: x[0])
    records = [r[1] for r in results]

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"完成。共 {total} 条，已写入 {args.output_file}")


if __name__ == "__main__":
    main()

