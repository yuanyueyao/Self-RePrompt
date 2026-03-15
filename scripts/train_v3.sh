#!/bin/bash
# v3 训练：合并 HotpotQA + GSM8K + OpenBookQA，过滤 misleading 样本
#
# 数据集统计（过滤 misleading 后）：
#   hotpot:     1887 条
#   gsm8k:      7390 条
#   openbookqa: 4533 条
#   合计:      13810 条
#
# 训练配置：
#   7 卡 (GPU 1-7) × batch 2 × grad_acc 4 = 有效 batch 56
#   3 epochs ≈ 740 步 / epoch，共约 2220 步
#
# 用法：
#   bash scripts/train_v3.sh
#   或指定 GPU：GPUS=0,1,2,3 bash scripts/train_v3.sh

set -e
cd "$(dirname "$0")/.."

GPUS=${GPUS:-"1,2,3,4,5,6,7"}
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

TRAIN_FILES=(
    "data/srp_prompt_with_answer/hotpot_train_qa_2000_with_srp_answer.jsonl"
    "data/srp_prompt_with_answer/gsm8k_train_with_srp_answer.jsonl"
    "data/srp_prompt_with_answer/openbookqa_train_with_srp_answer.jsonl"
)
TRAIN_FILE=$(IFS=,; echo "${TRAIN_FILES[*]}")

echo "=========================================="
echo "  Self-RePrompt v3 训练"
echo "  GPUs: $GPUS  (共 $N_GPUS 卡)"
echo "  数据: $(echo $TRAIN_FILE | tr ',' '\n' | wc -l) 个文件"
echo "  输出: outputs/qwen3_sr_lora_v3"
echo "=========================================="

mkdir -p logs

CUDA_VISIBLE_DEVICES=$GPUS \
torchrun \
    --nproc_per_node=$N_GPUS \
    --master_port=29503 \
    src/student/train_qwen3_sr_lora.py \
    --model_name_or_path model/Qwen3-8B \
    --train_file "$TRAIN_FILE" \
    --filter_quadrant "misleading" \
    --output_dir outputs/qwen3_sr_lora_v3 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --logging_steps 20 \
    --save_steps 500 \
    --save_total_limit 3 \
    --bf16 \
    --mask_user \
    2>&1 | tee logs/train_v3.log

echo "训练完成，adapter 已保存至 outputs/qwen3_sr_lora_v3"