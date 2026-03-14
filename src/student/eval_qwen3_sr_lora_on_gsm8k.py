import argparse
from pathlib import Path
from typing import Dict, List

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference of Qwen3-8B + SR-LoRA on GSM8K reprompt data."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="model/Qwen3-8B",
        help="Base Qwen3-8B model name or path.",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="Directory of LoRA adapter (output_dir from training).",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/srp_prompt/gsm8k_train_reprompt.jsonl",
        help="JSONL with {user, sr_prompt, answer}.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="Maximum number of samples for demo.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate.",
    )
    return parser.parse_args()


def load_jsonl(path: str, max_samples: int) -> List[Dict[str, str]]:
    data: List[Dict[str, str]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            user = (obj.get("user") or "").strip()
            answer = (obj.get("answer") or "").strip()
            if not user or not answer:
                continue
            data.append({"user": user, "answer": answer})
            if len(data) >= max_samples:
                break
    return data


def main() -> None:
    args = parse_args()

    # 从 adapter 目录加载 tokenizer（含 <SRP_START>/<SRP_END> 特殊 token）
    tokenizer = AutoTokenizer.from_pretrained(
        args.lora_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # 必须在加载 adapter 前将 base model 词表 resize 到 adapter 保存时的大小，
    # 否则 embed_tokens/lm_head shape 不匹配会报错
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()

    data = load_jsonl(args.data_file, args.max_samples)
    print(f"Loaded {len(data)} samples from {args.data_file}")

    for idx, sample in enumerate(data, start=1):
        user = sample["user"]
        messages = [
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)

        print(f"\n===== Sample {idx} =====")
        print("User question:")
        print(user)
        print("\nModel output:")
        print(decoded)
        print("\nGold answer:")
        print(sample["answer"])


if __name__ == "__main__":
    main()

