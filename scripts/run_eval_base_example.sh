#!/bin/bash
# 在 Qwen3-8B-Base + Base 上训练的 LoRA 上跑评测（示例）
# 前置：已完成 download_qwen3_8b_base.py + train_v3_base.sh
#
# 默认 8 卡并行（eval_lora_accuracy.py 按样本分片，约快 ~8 倍）：
#   bash scripts/run_eval_base_example.sh
#
# 指定 GPU 列表（逗号分隔，与机器上可见 GPU 编号一致）：
#   GPUS=0,1,2,3 bash scripts/run_eval_base_example.sh
#
# 单卡：
#   GPUS=0 bash scripts/run_eval_base_example.sh
#
# 参数：$1=dataset（默认 gsm8k） $2=max_samples（默认 200）

set -e
cd "$(dirname "$0")/.."

BASE="model/Qwen3-8B-Base"
LORA="outputs/qwen3_sr_lora_v3_base"
# 默认 8 张卡；若机器不足 8 张，请设置 GPUS=0,1,... 或单卡 GPUS=0
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

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

GPU_ARGS=()
if [ -n "$GPUS" ]; then
    GPU_ARGS=(--gpus "$GPUS")
fi

echo ">>> eval_lora_accuracy  dataset=$DS  samples=$SAMPLES  gpus=$GPUS"
python -u src/student/eval_lora_accuracy.py \
    --base_model "$BASE" \
    --lora_dir "$LORA" \
    --dataset "$DS" \
    --max_samples "$SAMPLES" \
    "${GPU_ARGS[@]}"
