import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用大语言模型为 Hotpot QA 样本生成 Self-RePrompt 数据（user/sr_prompt/answer 三元组）。"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        default="data/hotpot_train_qa_2000.jsonl",
        help="输入 Hotpot QA JSON 文件路径，格式为 [{\"question\": ..., \"answer\": ...}, ...]。",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        default="data/hotpot_train_qa_2000_reprompt_v2.jsonl",
        help="输出 JSONL 文件路径，每行一个 {user, sr_prompt, answer}。",
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
        default="deepseek-ai/DeepSeek-V3.2",
        help="teacher 大模型名称（示例使用 OpenAI 风格，可按需替换）。",
    )
    return parser.parse_args()


def build_teacher_messages(question: str) -> List[Dict[str, str]]:
    """
    Teacher 只看到 question，基于问题本身设计一条指导性指令（不提供答案）。
    """
    system_prompt = (
        "You are an expert prompt engineer.\n\n"

        "You will be given a user question. Your task is to write a SHORT guiding "
        "instruction that helps another LLM answer the question.\n\n"

        "Rules:\n"
        "- Language: English\n"
        "- Maximum 2 sentences\n"
        "- Maximum 40 words\n"
        "- Do NOT restate or paraphrase the question\n"
        "- Do NOT mention specific entities from the question\n"
        "- Describe only the reasoning strategy (e.g., identify entities, compare dates, follow relationships)\n"
        "- No explanations, no step-by-step paragraphs\n"
        "- Output ONLY the instruction text."
    )

    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Write the guiding instruction (as described above) that another model should follow to answer this question. "
        "Output ONLY the instruction text."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llm(model: str, messages: List[Dict[str, str]]) -> str:
    """
    调用外部大模型生成 sr_prompt。

    这里使用 SiliconFlow 的 OpenAI 兼容 API：
      - 从环境变量 SILICONFLOW_API_KEY 读取密钥
      - 从环境变量 SILICONFLOW_BASE_URL 读取 base_url（可选，默认 https://api.siliconflow.cn/v1）
    """
    api_key = os.getenv("SILICONFLOW_API_KEY","sk-efwradynmcooxyiglmldbxlhlkxecwjjgkcmcgdgtfnkazxr")
    base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

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
        data: List[Dict[str, Any]] = json.load(f)

    if args.max_samples is not None:
        data = data[: args.max_samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for item in data:
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            if not question or not answer:
                continue

            messages = build_teacher_messages(question)
            sr_prompt = call_llm(args.model, messages)

            record = {
                "user": question,
                "sr_prompt": sr_prompt,
                "answer": answer,
            }
            json.dump(record, out_f, ensure_ascii=False)
            out_f.write("\n")

            processed += 1
            if processed % 50 == 0:
                print(f"processed {processed} samples...")

    print(f"done. total processed: {processed}, output -> {output_path}")


if __name__ == "__main__":
    main()
