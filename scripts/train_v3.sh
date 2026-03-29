#!/bin/bash
# v3 训练：合并 HotpotQA + GSM8K + OpenBookQA
# 默认基座：Qwen3-8B-Base（预训练，非 Instruct）
#
# 默认过滤 quadrant：misleading + both_wrong
# 覆盖示例：FILTER_QUADRANT=misleading bash scripts/train_v3.sh
#
# 前置：python scripts/download_qwen3_8b_base.py
# 输出：outputs/qwen3_sr_lora_v3_base（与 Instruct 上训练的 adapter 不可互换）
#
# 用法：
#   bash scripts/train_v3.sh
#   GPUS=0,1,2,3 bash scripts/train_v3.sh
#   OUT_DIR=outputs/my_run MODEL_PATH=model/Qwen3-8B-Base bash scripts/train_v3.sh

set -e
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-model/Qwen3-8B-Base}"
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "错误: 未找到 $MODEL_PATH，请先执行: python scripts/download_qwen3_8b_base.py"
    exit 1
fi

GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

TRAIN_FILES=(
    "data/srp_prompt_with_answer/hotpot_train_qa_2000_with_srp_answer.jsonl"
    "data/srp_prompt_with_answer/gsm8k_train_with_srp_answer.jsonl"
    "data/srp_prompt_with_answer/openbookqa_train_with_srp_answer.jsonl"
)
TRAIN_FILE=$(IFS=,; echo "${TRAIN_FILES[*]}")

OUT_DIR="${OUT_DIR:-outputs/qwen3_sr_lora_v3_base}"
FILTER_QUADRANT="${FILTER_QUADRANT:-misleading,both_wrong}"

echo "=========================================="
echo "  Self-RePrompt v3 训练（Qwen3-8B-Base）"
echo "  基座: $MODEL_PATH"
echo "  GPUs: $GPUS  (共 $N_GPUS 卡)"
echo "  数据: $(echo $TRAIN_FILE | tr ',' '\n' | wc -l) 个文件"
echo "  输出: $OUT_DIR"
echo "  过滤 quadrant: $FILTER_QUADRANT"
echo "=========================================="

mkdir -p logs

CUDA_VISIBLE_DEVICES=$GPUS \
torchrun \
    --nproc_per_node=$N_GPUS \
    --master_port=29503 \
    src/student/train_qwen3_sr_lora.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --filter_quadrant "$FILTER_QUADRANT" \
    --output_dir "$OUT_DIR" \
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

echo "训练完成，adapter: $OUT_DIR"
