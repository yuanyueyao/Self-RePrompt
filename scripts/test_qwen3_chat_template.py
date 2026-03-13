from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # 或你实际要用的 Qwen3 / Qwen2 模型名


def main() -> None:
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    example = {
        "user": (
            "Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether "
            "in April and May?"
        ),
        "sr_prompt": (
            "Find half of the April sales to get May sales, then add the two "
            "monthly amounts together."
        ),
        "answer": "72",
    }

    assistant_content = (
        "<SRP_START>\n"
        + example["sr_prompt"]
        + "\n<SRP_END>\n"
        + example["answer"]
    )

    # 训练场景：user + assistant（含 SR 段 + 答案）
    train_messages = [
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": assistant_content},
    ]
    train_text = tokenizer.apply_chat_template(
        train_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 推理场景：仅给出 user，由模型生成 assistant
    infer_messages = [
        {"role": "user", "content": example["user"]},
    ]
    infer_text = tokenizer.apply_chat_template(
        infer_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("===== TRAIN TEMPLATE TEXT =====")
    print(train_text)
    print("===== INFER TEMPLATE TEXT =====")
    print(infer_text)


if __name__ == "__main__":
    main()

