"""
Self-RePrompt 项目 Qwen3-8B LoRA 训练脚本。

功能概览：
1. 从 JSONL 数据文件中读取 Self-RePrompt 三元组样本：
   - 每行包含字段：{"user": "...", "sr_prompt": "...", "srp_answer": "..."}。

2. 将每条样本拼接为单轮对话训练文本：
   - 使用 Qwen3 自带的 chat_template（apply_chat_template）构造；
   - user role：原始题目/指令；
   - assistant role：<SRP_START>{sr_prompt}<SRP_END>{srp_answer}；
   - <SRP_START> / <SRP_END> 标记“内部自重写 prompt 段”，用于训练模型先写内部 prompt 再回答。

3. 基于 HuggingFace Transformers + PEFT：
   - 从 --model_name_or_path 加载 Qwen3-8B 基座模型与 tokenizer（支持 HF 模型名或本地路径如 model/Qwen3-8B）；
   - 为 tokenizer 注册 <SRP_START> / <SRP_END> 特殊 token，并调用 resize_token_embeddings；
   - 使用 LoRA 在注意力层 q_proj/k_proj/v_proj/o_proj 上挂载可训练低秩矩阵。

4. 训练流程：
   - 将 JSONL 数据构造成 Dataset，并按 95%/5% 划分 train/dev；
   - 对每条样本做 tokenize，长度截断为 --max_seq_length；
   - 对整段文本做自回归交叉熵 loss；
   - 若指定 --mask_user，则对 <assistant> 之前的 token label 置为 -100（通过 apply_chat_template 精确定位边界）；
   - 使用 Trainer 封装训练循环，支持多卡（通过 accelerate/torchrun 启动）、断点续训等。

5. 产出：
   - 在 --output_dir 下保存 LoRA adapter 权重（PeftModel.save_pretrained）；
   - 同时保存包含新增特殊 token 的 tokenizer，方便推理脚本直接加载。

典型用法示例（7 卡 torchrun）：

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 src/student/train_qwen3_sr_lora.py \\
        --model_name_or_path model/Qwen3-8B \\
        --train_file data/srp_prompt_with_answer/hotpot_train_qa_2000_with_srp_answer.jsonl \\
        --output_dir outputs/qwen3_sr_lora_hotpot_v2 \\
        --max_seq_length 1024 \\
        --per_device_train_batch_size 2 \\
        --gradient_accumulation_steps 2 \\
        --num_train_epochs 3 \\
        --learning_rate 2e-4 \\
        --warmup_ratio 0.03 \\
        --bf16 \\
        --mask_user

搭配 eval_qwen3_sr_lora_on_gsm8k.py 等评估脚本，用于对比微调前后模型在 GSM8K 等任务上的表现差异。
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model


SPECIAL_TOKENS = {
    "sr_prompt_begin": "<SRP_START>",
    "sr_prompt_end": "<SRP_END>",
}


@dataclass
class TrainConfig:
    model_name_or_path: str
    train_file: str
    output_dir: str
    filter_quadrant: str = "misleading"
    max_seq_length: int = 1024
    mask_user: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 20
    save_steps: int = 1000
    save_total_limit: int = 3
    bf16: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA finetune Qwen3-8B for Self-RePrompt on srp_prompt data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="model/Qwen3-8B",
        help=(
            "基座路径：Instruct 用 model/Qwen3-8B 或 Qwen/Qwen3-8B；"
            "预训练底座用 model/Qwen3-8B-Base 或 Qwen/Qwen3-8B-Base（需单独训练 LoRA，不可混用 adapter）。"
        ),
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=(
            "data/srp_prompt_with_answer/hotpot_train_qa_2000_with_srp_answer.jsonl,"
            "data/srp_prompt_with_answer/gsm8k_train_with_srp_answer.jsonl,"
            "data/srp_prompt_with_answer/openbookqa_train_with_srp_answer.jsonl"
        ),
        help="JSONL 文件路径，多个文件用逗号分隔，会合并为一个训练集。",
    )
    parser.add_argument(
        "--filter_quadrant",
        type=str,
        default="misleading",
        help="逗号分隔的 quadrant 值，这些样本将被过滤掉。"
             "默认过滤 'misleading'（sr_prompt 误导模型的样本）。"
             "设为空字符串则不过滤。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen3_sr_lora_v3",
        help="Directory to save LoRA adapter and tokenizer.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--mask_user",
        action="store_true",
        help="Mask <USER> part from loss (labels=-100).",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=20,
        help="Logging steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Checkpoint save steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 training.",
    )
    return parser.parse_args()


def load_jsonl_dataset(train_files: str, filter_quadrants: set) -> Dataset:
    """
    加载一个或多个 JSONL 文件并合并为单个 Dataset。

    Args:
        train_files: 逗号分隔的文件路径字符串。
        filter_quadrants: 需要过滤掉的 quadrant 值集合（如 {"misleading"}）。
    """
    paths = [p.strip() for p in train_files.split(",") if p.strip()]
    records: List[Dict[str, str]] = []
    stats: Dict[str, Dict] = {}

    for path in paths:
        p = Path(path)
        file_records = 0
        file_filtered = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                user = (obj.get("user") or "").strip()
                sr_prompt = (obj.get("sr_prompt") or "").strip()
                srp_answer = (obj.get("srp_answer") or "").strip()
                if not user or not srp_answer or not sr_prompt:
                    continue
                # quadrant 过滤
                quadrant = obj.get("quadrant", "")
                if quadrant in filter_quadrants:
                    file_filtered += 1
                    continue
                records.append({"user": user, "sr_prompt": sr_prompt, "srp_answer": srp_answer})
                file_records += 1
        stats[p.name] = {"kept": file_records, "filtered": file_filtered}

    if not records:
        raise RuntimeError(f"No valid records found in: {train_files}")

    print(f"\n数据集加载汇总（过滤 quadrant: {filter_quadrants or '无'}）：")
    total_kept = 0
    for name, s in stats.items():
        total_kept += s["kept"]
        print(f"  {name}: {s['kept']} 条保留，{s['filtered']} 条过滤")
    print(f"  合计: {total_kept} 条训练样本\n")

    return Dataset.from_list(records)


def build_sample_text(example: Dict[str, str]) -> str:
    # NOTE: 仅保留 SRP 标记，角色格式交给 Qwen3 自带的 chat template 处理。
    return (
        f"{SPECIAL_TOKENS['sr_prompt_begin']}\n"
        f"{example['sr_prompt']}\n"
        f"{SPECIAL_TOKENS['sr_prompt_end']}\n"
        f"{example['srp_answer']}"
    )


def add_special_tokens(tokenizer: AutoTokenizer) -> AutoTokenizer:
    special_tokens = []
    for v in SPECIAL_TOKENS.values():
        if v not in tokenizer.get_vocab():
            special_tokens.append(v)
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer


def create_model_and_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = add_special_tokens(tokenizer)

    # torchrun DDP：每个进程只占用自己对应的 GPU，避免 device_map="auto" 与 DDP 冲突
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map={"": local_rank},
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # 确保新增 token 的 embedding 和 lm_head 被训练并保存进 adapter
        modules_to_save=["embed_tokens", "lm_head"],
        # Qwen3 tie_word_embeddings=True：保持 embed_tokens 与 lm_head 权重同步，
        # 避免训练时两者独立更新，也可防止 adapter 体积翻倍
        ensure_weight_tying=True,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def tokenize_function(
    example: Dict[str, str],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    mask_user: bool,
) -> Dict[str, List[int]]:
    # 使用 Qwen3 自带 chat_template 构造单轮对话：
    # - user role：原始题目/指令
    # - assistant role：显式包含 <SRP_START> ... <SRP_END> + 最终答案
    assistant_content = build_sample_text(example)
    messages = [
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": assistant_content},
    ]
    # add_generation_prompt=False，表示这是完整的 user+assistant 监督数据
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # 只做截断，不做 padding（padding 由 data collator 按 batch 内最长动态填充）
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = tokenized["input_ids"]
    labels = list(input_ids)  # 有效 token 全部监督

    if mask_user:
        # 只对 assistant 输出部分计算 loss：用 add_generation_prompt=True 获取 user 前缀长度
        user_prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["user"]}],
            tokenize=False,
            add_generation_prompt=True,  # 包含 <|im_start|>assistant\n
        )
        user_prefix_len = len(tokenizer(user_prefix, add_special_tokens=False)["input_ids"])
        for i in range(min(user_prefix_len, len(labels))):
            labels[i] = -100

    tokenized["labels"] = labels
    return tokenized


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        filter_quadrant=args.filter_quadrant,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        mask_user=args.mask_user,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
    )

    filter_set = {q.strip() for q in cfg.filter_quadrant.split(",") if q.strip()}
    raw_dataset = load_jsonl_dataset(cfg.train_file, filter_set)
    split = raw_dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    model, tokenizer = create_model_and_tokenizer(cfg)

    tokenized_train = train_ds.map(
        lambda ex: tokenize_function(ex, tokenizer, cfg.max_seq_length, cfg.mask_user),
        remove_columns=train_ds.column_names,
    )
    tokenized_eval = eval_ds.map(
        lambda ex: tokenize_function(ex, tokenizer, cfg.max_seq_length, cfg.mask_user),
        remove_columns=eval_ds.column_names,
    )

    # 动态 padding：每个 batch 只 pad 到该 batch 内最长序列，labels 用 -100 填充
    # pad_to_multiple_of=8 让序列长度对齐 8 的倍数，更好利用 GPU Tensor Core
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        eval_steps=cfg.save_steps,
        bf16=cfg.bf16,
        report_to=["none"],
        # modules_to_save 的 ModulesToSaveWrapper 在 DDP 下会被误判为 unused，关闭检查
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()

    save_dir = Path(cfg.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()

