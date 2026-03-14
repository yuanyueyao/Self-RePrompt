"""
train_qwen3_sr_lora.py 阶段性诊断脚本。

测试项：
  1. Tokenizer & 特殊 Token  —— 词表扩展 +2、每个特殊 token 编码为单个 ID
  2. Chat Template 格式      —— 多角色对话文本格式、字段顺序
  3. Tokenize Pipeline       —— padding / labels / mask_user 是否正确标注
  4. 模型可训练性（可选）    —— embed_tokens / lm_head / LoRA 层是否 requires_grad

用法：
    # 仅测试 tokenizer 和数据管道（无需 GPU，几秒内完成）
    python src/student/test_train_pipeline.py --skip_model

    # 完整测试（含模型加载，需要 GPU，约 1-2 分钟）
    python src/student/test_train_pipeline.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_qwen3_sr_lora import (
    SPECIAL_TOKENS,
    TrainConfig,
    add_special_tokens,
    build_sample_text,
    tokenize_function,
    create_model_and_tokenizer,
)
from transformers import AutoTokenizer


PASS = "✅ PASS"
FAIL = "❌ FAIL"

_all_pass = True


def check(cond: bool, msg: str) -> bool:
    global _all_pass
    if not cond:
        _all_pass = False
    print(f"  {PASS if cond else FAIL}  {msg}")
    return cond


def section(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────
# 1. Tokenizer & 特殊 Token
# ─────────────────────────────────────────────────────────────
def test_tokenizer(model_path: str) -> AutoTokenizer:
    section("1. Tokenizer & 特殊 Token")

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    vocab_before = len(tok)
    print(f"  词表大小（添加前）: {vocab_before}")

    tok = add_special_tokens(tok)
    vocab_after = len(tok)
    print(f"  词表大小（添加后）: {vocab_after}")

    check(vocab_after == vocab_before + 2, f"词表扩展 +2（{vocab_before} → {vocab_after}）")

    for _key, token_str in SPECIAL_TOKENS.items():
        ids = tok.encode(token_str, add_special_tokens=False)
        check(
            len(ids) == 1,
            f"{token_str!r} 编码为单个 token，ID={ids[0] if ids else 'N/A'}（共 {len(ids)} 个）",
        )
        all_special = getattr(tok, "additional_special_tokens", None) or tok.all_special_tokens
        check(token_str in all_special, f"{token_str!r} 在 special_tokens 列表中")

        # 反向验证：decode 回来应仍是原字符串
        decoded = tok.decode([ids[0]])
        check(decoded == token_str, f"decode({ids[0]}) == {token_str!r}（实际: {decoded!r}）")

    return tok


# ─────────────────────────────────────────────────────────────
# 2. Chat Template 格式
# ─────────────────────────────────────────────────────────────
def test_chat_template(tok: AutoTokenizer) -> None:
    section("2. Chat Template 格式")

    fake = {
        "user": "What is the capital of France?",
        "sr_prompt": "Think step by step about geography.",
        "srp_answer": "The capital of France is Paris.",
    }

    assistant_content = build_sample_text(fake)
    messages = [
        {"role": "user", "content": fake["user"]},
        {"role": "assistant", "content": assistant_content},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    print("\n  ── 完整对话文本 ──")
    print(text)
    print("  ── end ──\n")

    check("<|im_start|>" in text, "包含 Qwen3 角色标记 <|im_start|>")
    check("user" in text and "assistant" in text, "包含 user / assistant 角色")
    check(SPECIAL_TOKENS["sr_prompt_begin"] in text, f"包含 {SPECIAL_TOKENS['sr_prompt_begin']}")
    check(SPECIAL_TOKENS["sr_prompt_end"] in text, f"包含 {SPECIAL_TOKENS['sr_prompt_end']}")
    check(fake["user"] in text, "user 问题内容完整")
    check(fake["srp_answer"] in text, "srp_answer 内容完整")

    pos_start = text.find(SPECIAL_TOKENS["sr_prompt_begin"])
    pos_end = text.find(SPECIAL_TOKENS["sr_prompt_end"])
    pos_ans = text.find(fake["srp_answer"])
    check(0 <= pos_start < pos_end, "SRP_START 在 SRP_END 之前")
    check(pos_end < pos_ans, "srp_answer 在 SRP_END 之后")

    # 验证不包含生成提示符（训练时应用完整 user+assistant 文本）
    user_prefix = tok.apply_chat_template(
        [{"role": "user", "content": fake["user"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_len = len(tok(user_prefix, add_special_tokens=False)["input_ids"])
    full_len = len(tok(text, add_special_tokens=False)["input_ids"])
    check(full_len > prefix_len, f"完整序列长度({full_len}) > user 前缀长度({prefix_len})，assistant 部分存在")


# ─────────────────────────────────────────────────────────────
# 3. Tokenize Pipeline（attention_mask / labels）
# ─────────────────────────────────────────────────────────────
def test_tokenize_pipeline(tok: AutoTokenizer, max_seq_length: int = 128) -> None:
    section("3. Tokenize Pipeline（attention_mask / labels）")

    fake = {
        "user": "What is the capital of France?",
        "sr_prompt": "Think step by step.",
        "srp_answer": "Paris.",
    }

    for mask_user in (False, True):
        print(f"\n  ── mask_user={mask_user} ──")
        result = tokenize_function(fake, tok, max_seq_length, mask_user=mask_user)

        ids    = result["input_ids"]
        mask   = result["attention_mask"]
        labels = result["labels"]

        check(len(ids)    == max_seq_length, f"input_ids    长度 == {max_seq_length}")
        check(len(mask)   == max_seq_length, f"attention_mask 长度 == {max_seq_length}")
        check(len(labels) == max_seq_length, f"labels       长度 == {max_seq_length}")

        pad_pos  = [i for i, m in enumerate(mask) if m == 0]
        real_pos = [i for i, m in enumerate(mask) if m == 1]
        print(f"  有效 token: {len(real_pos)}，padding token: {len(pad_pos)}")

        if pad_pos:
            check(all(labels[i] == -100 for i in pad_pos), "所有 padding 位置 label == -100")
        else:
            print(f"  [INFO] 序列未被 padding（max_seq_length={max_seq_length} 可能偏小）")

        if real_pos:
            has_signal = any(labels[i] != -100 for i in real_pos)
            check(has_signal, "有效 token 中存在 label != -100（监督信号）")

        if mask_user and real_pos:
            masked_in_real = sum(1 for i in real_pos if labels[i] == -100)
            total_real = len(real_pos)
            print(f"  有效区间内被 mask token: {masked_in_real}/{total_real}（user 前缀）")
            check(masked_in_real > 0,          "mask_user=True：有效区间内有被 mask 的 token")
            check(masked_in_real < total_real, "mask_user=True：assistant 部分仍保留监督信号")

        # 展示被监督的 token 文本，便于直觉验证
        supervised_ids = [ids[i] for i in real_pos if labels[i] != -100]
        supervised_text = tok.decode(supervised_ids, skip_special_tokens=False)
        print(f"  监督文本（前 200 字）: {repr(supervised_text[:200])}")

        # 额外检查：SRP_START / SRP_END 必须出现在监督段中
        srp_start_id = tok.convert_tokens_to_ids(SPECIAL_TOKENS["sr_prompt_begin"])
        srp_end_id   = tok.convert_tokens_to_ids(SPECIAL_TOKENS["sr_prompt_end"])
        supervised_set = set(supervised_ids)
        check(srp_start_id in supervised_set, f"SRP_START (id={srp_start_id}) 在监督 token 中")
        check(srp_end_id   in supervised_set, f"SRP_END   (id={srp_end_id})   在监督 token 中")


# ─────────────────────────────────────────────────────────────
# 4. 模型可训练性（需要 GPU）
# ─────────────────────────────────────────────────────────────
def test_model_trainability(model_path: str) -> None:
    import torch

    section("4. 模型可训练性（embed_tokens / lm_head / LoRA）")

    cfg = TrainConfig(
        model_name_or_path=model_path,
        train_file="",
        output_dir="",
        bf16=torch.cuda.is_bf16_supported(),
    )
    model, _tok = create_model_and_tokenizer(cfg)
    model.print_trainable_parameters()

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

    embed_trainable = [n for n in trainable_names if "embed_tokens" in n]
    lmhead_trainable = [n for n in trainable_names if "lm_head" in n]
    lora_trainable   = [n for n in trainable_names if "lora_" in n]

    check(len(embed_trainable) > 0, f"embed_tokens 可训练（{len(embed_trainable)} 个参数块）")
    check(len(lmhead_trainable) > 0, f"lm_head      可训练（{len(lmhead_trainable)} 个参数块）")
    check(len(lora_trainable)   > 0, f"LoRA 层      可训练（{len(lora_trainable)} 个参数块）")

    print(f"\n  embed_tokens 参数块:\n    " + "\n    ".join(embed_trainable))
    print(f"\n  lm_head 参数块:\n    " + "\n    ".join(lmhead_trainable))


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train_qwen3_sr_lora.py 诊断脚本")
    parser.add_argument("--model_name_or_path", default="model/Qwen3-8B")
    parser.add_argument("--skip_model", action="store_true", help="跳过模型加载（无 GPU 时使用）")
    parser.add_argument("--max_seq_length", type=int, default=128, help="tokenize 测试用的序列长度")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tok = test_tokenizer(args.model_name_or_path)
    test_chat_template(tok)
    test_tokenize_pipeline(tok, max_seq_length=args.max_seq_length)

    if not args.skip_model:
        test_model_trainability(args.model_name_or_path)
    else:
        print("\n[跳过模型加载测试，去掉 --skip_model 可开启]")

    print("\n" + "=" * 62)
    if _all_pass:
        print("  ✅ 全部测试通过")
    else:
        print("  ❌ 存在失败项，请检查上方输出")
    print("=" * 62)


if __name__ == "__main__":
    main()
