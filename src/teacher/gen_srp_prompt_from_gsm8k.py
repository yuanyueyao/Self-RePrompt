"""
使用大语言模型为 GSM8K 样本生成 Self-RePrompt 训练用 sr_prompt（user/sr_prompt/answer 三元组）。

使用方式:
    # 默认：train 子集，全部样本
    python src/gen_srp_prompt_from_gsm8k.py

    # 快速测试 10 条
    python src/gen_srp_prompt_from_gsm8k.py --max_samples 10

    # 使用 test 子集
    python src/gen_srp_prompt_from_gsm8k.py --split test --output data/reprompt_reason/gsm8k_test_reprompt.jsonl

    # 启用 solution hint（将 answer_full 中 #### 前的解题步骤传给 teacher）
    python src/gen_srp_prompt_from_gsm8k.py --use_solution --max_samples 5
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用大语言模型为 GSM8K 样本生成 Self-RePrompt 数据（user/sr_prompt/answer 三元组）。"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default="data/raw/gsm8k.json",
        help="输入 GSM8K JSON 文件路径，格式为 {\"train\": [...], \"test\": [...]}。",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="data/srp_prompt/gsm8k_train_reprompt.jsonl",
        help="输出 JSONL 文件路径，每行一个 {user, sr_prompt, answer}。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="使用 train 或 test 子集。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多处理多少条样本（用于快速试验）。默认处理全部。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="DeepSeek 官方模型名称（chat/completion 模型）。",
    )
    parser.add_argument(
        "--use_solution",
        action="store_true",
        help="若数据含 answer_full，将 #### 前的解题步骤作为 hint 传给 teacher。",
    )
    return parser.parse_args()


def build_teacher_messages(
    question: str,
    solution_hint: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Teacher 基于 question（及可选的 solution_hint）设计指导性指令。
    solution_hint 为 answer_full 中 #### 之前的部分，提供解题结构但不含最终答案。
    """
    system_prompt = (
        "You are an expert prompt engineer for math word problems.\n\n"

        "You will be given a math word problem. Your task is to write a SHORT guiding "
        "instruction that helps another LLM solve the problem correctly.\n\n"

        "Rules:\n"
        "- Language: English\n"
        "- Maximum 2 sentences, up to 50 words\n"
        "- Do NOT restate or paraphrase the problem\n"
        "- Do NOT mention specific numbers from the problem\n"
        "- Describe only the reasoning strategy (how to approach, not the answer)\n"
        "- No explanations, no step-by-step paragraphs\n"
        "- Output ONLY the instruction text\n"
        "- Avoid ambiguous terms that can mislead (see pitfalls below)\n\n"

        "Strategy types (pick the most relevant):\n"
        "- Unit conversion: convert rates (e.g., per hour to per minute) before multiplying\n"
        "- Ratio/proportion: find per-part value, then scale to each share\n"
        "- Sequential changes: trace state step by step (triple, add, double, etc.)\n"
        "- Natural language relations: parse phrases like 'N less than half as many' as (half of X) - N\n"
        "- Multi-step: identify intermediate quantities and solve in dependency order\n"
        "- Round-trip / both directions: 'to and from' means multiply one-way time by 2 per trip\n"
        "- Watch for: unit confusion (min vs hour), 'twice as many' vs 'two more than'\n\n"

        "Pitfalls to avoid (use precise wording):\n"
        "- For ratio problems: use 'ratio of A to total' or 'A per total unit', NOT 'A-to-B' when B is ambiguous (e.g., 'tea-to-water' can confuse—use 'tea per cup' or 'tea-to-mixture')\n"
        "- For sequential removals: say 'update remaining count after EACH removal before summing' to avoid forgetting a step\n"
        "- For profit with one-time cost: explicitly mention 'subtract the one-time cost at the end'\n\n"

        "Examples of good instructions:\n\n"

        "Problem: Weng earns $12 an hour. She did 50 minutes of babysitting. How much did she earn?\n"
        "Instruction: Convert the hourly rate to per-minute rate, then multiply by minutes worked.\n\n"

        "Problem: Roque walks 2h one way to work, bikes 1h one way. He walks to-and-from 3 times/week, bikes 2 times/week. Total hours?\n"
        "Instruction: Each 'to and from' is a round trip (double the one-way time). Multiply round-trip time by weekly frequency for each mode, then sum.\n\n"

        "Problem: 8-oz cup uses 1 oz tea. Same ratio for 12 people, 6 oz each. How many oz tea needed?\n"
        "Instruction: Find tea per total cup (1/8) from the example; multiply total ounces of mixture needed by that fraction.\n\n"

        "Problem: Weight starts at 2 lb, then triples, then add 2 lb, then doubles. Final weight?\n"
        "Instruction: Apply each operation in sequence: multiply by 3, add 2, then multiply by 2.\n\n"

        "Problem: Carl takes 4 pink hats, John takes 6 pink and twice as many green. Remaining hats?\n"
        "Instruction: Update pink after Carl, then update pink and green after John; sum all remaining."
    )

    user_content = "Math problem:\n" f"{question}\n\n"
    if solution_hint:
        user_content += (
            "Solution structure (for reference only; do NOT copy or reveal the answer):\n"
            f"{solution_hint[:500]}\n\n"
        )
    user_content += (
        "Write the guiding instruction (as described above) that another model should follow. "
        "Output ONLY the instruction text."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def call_llm(model: str, messages: List[Dict[str, str]]) -> str:
    """
    调用 DeepSeek 官方 OpenAI SDK 生成 sr_prompt。

    - 从环境变量 DEEPSEEK_API_KEY 读取密钥
    - 使用官方 base_url https://api.deepseek.com
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请先在环境变量 DEEPSEEK_API_KEY 中配置 DeepSeek API 密钥")
    base_url = "https://api.deepseek.com"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, List[Dict[str, Any]]] = json.load(f)

    data = raw.get(args.split, [])
    if not data:
        raise SystemExit(f"Split '{args.split}' not found or empty in {input_path}")

    if args.max_samples is not None:
        data = data[: args.max_samples]

    total = len(data)
    print(f"加载数据：{input_path} split={args.split}，共 {total} 条样本")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for item in data:
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            if not question or not answer:
                continue

            solution_hint = None
            if args.use_solution:
                full = (item.get("answer_full") or "").strip()
                if "####" in full:
                    solution_hint = full.split("####")[0].strip()

            messages = build_teacher_messages(question, solution_hint)
            sr_prompt = call_llm(args.model, messages)

            record = {
                "user": question,
                "sr_prompt": sr_prompt,
                "answer": answer,
            }
            json.dump(record, out_f, ensure_ascii=False)
            out_f.write("\n")

            processed += 1
            if processed % 10 == 0 or processed == total:
                pct = processed / total * 100
                print(f"[GSM8K SRP_PROMPT] {processed}/{total} ({pct:.1f}%)")

    print(f"done. total processed: {processed}, output -> {output_path}")


if __name__ == "__main__":
    main()