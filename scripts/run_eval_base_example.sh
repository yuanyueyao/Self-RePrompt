#!/bin/bash
# 在 Qwen3-8B-Base + Base 上训练的 LoRA 上跑评测（示例）
# 前置：已完成 download_qwen3_8b_base.py + train_v3_base.sh
#
# 用法：
#   bash scripts/run_eval_base_example.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_eval_base_example.sh

set -e
cd "$(dirname "$0")/.."

BASE="model/Qwen3-8B-Base"
LORA="outputs/qwen3_sr_lora_v3_base"

if [ ! -f "$BASE/config.json" ]; then
    echo "缺少 $BASE，请运行: python scripts/download_qwen3_8b_base.py"
    exit 1
fi
if [ ! -f "$LORA/adapter_config.json" ]; then
    echo "缺少 $LORA，请先训练: bash scripts/train_v3_base.sh"
    exit 1
fi

DS="${1:-gsm8k}"
SAMPLES="${2:-200}"

echo ">>> eval_lora_accuracy  dataset=$DS  samples=$SAMPLES"
python -u src/student/eval_lora_accuracy.py \
    --base_model "$BASE" \
    --lora_dir "$LORA" \
    --dataset "$DS" \
    --max_samples "$SAMPLES"
