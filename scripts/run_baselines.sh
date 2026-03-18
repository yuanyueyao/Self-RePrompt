#!/bin/bash
# 在三个数据集上跑所有 baseline 对比，分别占用 GPU 0/1/2
#
# 5 种条件：
#   B0  Base Qwen3-8B（无修改）                    ← 基准线
#   B1  Base + CoT/策略提示（零样本 prompt 工程）   ← 无需训练的对比
#   B2  Base + Oracle SRP（teacher sr_prompt 注入）← 效果上界
#   B4  SRP-LoRA v3（我们的方法）                  ← 本方法
#
# 用法：
#   bash scripts/run_baselines.sh
#   指定 GPU：GPU_HOTPOT=0 GPU_GSM8K=1 GPU_OBQA=2 bash scripts/run_baselines.sh

set -e
cd "$(dirname "$0")/.."

LORA=${LORA:-"outputs/qwen3_sr_lora_v3"}
SAMPLES=${SAMPLES:-200}
GPU_HOTPOT=${GPU_HOTPOT:-0}
GPU_GSM8K=${GPU_GSM8K:-1}
GPU_OBQA=${GPU_OBQA:-2}
MODES=${MODES:-"B0,B1,B2,B4"}

mkdir -p logs

echo "=========================================="
echo "  SRP Baseline 对比评测"
echo "  LoRA: $LORA"
echo "  样本数: $SAMPLES / 数据集"
echo "  模式: $MODES"
echo "=========================================="
echo ""

echo "[HotpotQA] GPU=$GPU_HOTPOT → logs/baseline_hotpot.log"
CUDA_VISIBLE_DEVICES=$GPU_HOTPOT python -u src/student/eval_baselines.py \
    --dataset hotpot \
    --lora_dir "$LORA" \
    --max_samples $SAMPLES \
    --modes "$MODES" \
    2>&1 | tee logs/baseline_hotpot.log &
PID_HOTPOT=$!

echo "[GSM8K]    GPU=$GPU_GSM8K → logs/baseline_gsm8k.log"
CUDA_VISIBLE_DEVICES=$GPU_GSM8K python -u src/student/eval_baselines.py \
    --dataset gsm8k \
    --lora_dir "$LORA" \
    --max_samples $SAMPLES \
    --modes "$MODES" \
    2>&1 | tee logs/baseline_gsm8k.log &
PID_GSM8K=$!

echo "[OpenBookQA] GPU=$GPU_OBQA → logs/baseline_openbookqa.log"
CUDA_VISIBLE_DEVICES=$GPU_OBQA python -u src/student/eval_baselines.py \
    --dataset openbookqa \
    --lora_dir "$LORA" \
    --max_samples $SAMPLES \
    --modes "$MODES" \
    2>&1 | tee logs/baseline_openbookqa.log &
PID_OBQA=$!

wait $PID_HOTPOT $PID_GSM8K $PID_OBQA

echo ""
echo "=========================================="
echo "  所有数据集评测完成！汇总如下："
echo "=========================================="
echo ""
echo "=== HotpotQA ==="
grep -E "^\s+(B[0-4]|={3,}|数据集)" logs/baseline_hotpot.log || true
echo ""
echo "=== GSM8K ==="
grep -E "^\s+(B[0-4]|={3,}|数据集)" logs/baseline_gsm8k.log || true
echo ""
echo "=== OpenBookQA ==="
grep -E "^\s+(B[0-4]|={3,}|数据集)" logs/baseline_openbookqa.log || true
