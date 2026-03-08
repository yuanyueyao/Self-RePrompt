"""
对比：原始问题 vs 改写后问题（sr_prompt）作为 user 消息的准确率。
用于 data/reprompt_question 数据：无系统提示，direct=仅原始问题作 user，sr=仅 sr_prompt（改写问题）作 user。
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比：原始问题 vs 改写问题(sr_prompt) 作为 user 消息的准确率（无 system 提示）。"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V3.2",
        help="用于评测的 API 模型名称。",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/reprompt_question/hotpot_train_qa_2000_reprompt_v0_rewrite_question.jsonl",
        help="包含 {user, sr_prompt, answer} 的 JSONL 文件路径（reprompt_question 数据）。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="最多评测多少条样本。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="每条样本生成的最大 new tokens。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行请求的线程数。",
    )
    return parser.parse_args()


def get_client() -> OpenAI:
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("请设置环境变量 SILICONFLOW_API_KEY")
    base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def load_data(path: str, max_samples: int) -> List[Dict[str, str]]:
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
            if len(data) >= max_samples:
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
    """无系统提示，仅原始问题作为 user 消息。"""
    return [{"role": "user", "content": user}]


def build_messages_with_sr(user: str, sr_prompt: str) -> List[Dict[str, str]]:
    """无系统提示，仅将 sr_prompt（改写后的问题）作为 user 消息。"""
    return [{"role": "user", "content": sr_prompt}]


def eval_once(
    client: OpenAI,
    model_name: str,
    sample: Dict[str, str],
    max_new_tokens: int,
) -> Tuple[bool, bool, str, str]:
    user = sample["user"]
    sr_prompt = sample["sr_prompt"]
    gold = sample["answer"]

    msg_direct = build_messages_direct(user)
    pred_direct = generate_answer_api(client, model_name, msg_direct, max_new_tokens)

    msg_sr = build_messages_with_sr(user, sr_prompt)
    pred_sr = generate_answer_api(client, model_name, msg_sr, max_new_tokens)

    ok_direct = answer_match(pred_direct, gold)
    ok_sr = answer_match(pred_sr, gold)
    return ok_direct, ok_sr, pred_direct, pred_sr


def _eval_one(
    client: OpenAI,
    model_name: str,
    max_new_tokens: int,
    idx: int,
    sample: Dict[str, str],
) -> Tuple[int, bool, bool, str, str, Dict[str, str]]:
    ok_direct, ok_sr, pred_direct, pred_sr = eval_once(
        client, model_name, sample, max_new_tokens
    )
    return idx, ok_direct, ok_sr, pred_direct, pred_sr, sample


def main() -> None:
    args = parse_args()
    client = get_client()

    print(f"加载数据：{args.data_file}（最多 {args.max_samples} 条）")
    data = load_data(args.data_file, args.max_samples)
    total = len(data)
    print(f"实际评测样本数：{total}，并行 workers={args.workers}")
    if total == 0:
        print("没有有效样本，退出。")
        return

    results: List[Tuple[int, bool, bool, str, str, Dict[str, str]]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _eval_one,
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
                print(f"已评测 {done}/{total} 条...")

    results.sort(key=lambda x: x[0])
    direct_correct = sum(1 for r in results if r[1])
    sr_correct = sum(1 for r in results if r[2])

    examples = []
    disagreements = []
    for r in results:
        idx, ok_direct, ok_sr, pred_direct, pred_sr, sample = r
        if ok_direct != ok_sr:
            disagreements.append(
                {
                    "user": sample["user"],
                    "gold": sample["answer"],
                    "pred_direct": pred_direct,
                    "pred_sr": pred_sr,
                    "ok_direct": ok_direct,
                    "ok_sr": ok_sr,
                }
            )
        if idx <= 5:
            examples.append(
                {
                    "user": sample["user"],
                    "gold": sample["answer"],
                    "pred_direct": pred_direct,
                    "pred_sr": pred_sr,
                    "ok_direct": ok_direct,
                    "ok_sr": ok_sr,
                }
            )

    direct_acc = direct_correct / total
    sr_acc = sr_correct / total

    print("\n===== 结果汇总 =====")
    print(f"总样本数: {total}")
    print(f"原始问题(user) 准确条数: {direct_correct}  准确率: {direct_acc:.3f}")
    print(f"改写问题(sr_prompt as user) 准确条数: {sr_correct}  准确率: {sr_acc:.3f}")

    print("\n===== 前几条示例（便于人工检查） =====")
    for i, ex in enumerate(examples, start=1):
        print(f"\n--- 样本 {i} ---")
        print(f"Q(原始): {ex['user']}")
        print(f"Gold: {ex['gold']}")
        print(f"[原始问题]  pred: {ex['pred_direct']!r}  ok={ex['ok_direct']}")
        print(f"[改写问题] pred: {ex['pred_sr']!r}  ok={ex['ok_sr']}")

    print("\n===== 结果不一致的样本（原始问题 vs 改写问题） =====")
    if not disagreements:
        print("两种方式在所有样本上的对错完全一致。")
    else:
        for i, ex in enumerate(disagreements, start=1):
            print(f"\n*** 不一致样本 {i} ***")
            print(f"Q: {ex['user']}")
            print(f"Gold: {ex['gold']}")
            print(f"[原始问题]  pred: {ex['pred_direct']!r}  ok={ex['ok_direct']}")
            print(f"[改写问题] pred: {ex['pred_sr']!r}  ok={ex['ok_sr']}")


if __name__ == "__main__":
    main()
