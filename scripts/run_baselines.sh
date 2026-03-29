#!/bin/bash
# 一次只评测一个数据集；对该次评测使用 GPUS 中的全部卡（eval_baselines.py 内多进程分片）。
#
# 5 种条件（默认 MODES=B0,B1,B2,B4）：
#   B0  Base Qwen3-8B-Base（无修改）
#   B1  Base + CoT/策略提示
#   B2  Base + Oracle SRP
#   B4  SRP-LoRA v3
#
# 用法：
#   bash scripts/run_baselines.sh hotpot      # 默认 8 卡 0–7 → logs/baseline_hotpot.log
#   bash scripts/run_baselines.sh gsm8k
#   bash scripts/run_baselines.sh openbookqa
#   bash scripts/run_baselines.sh super_gpqa | mmlu_pro | bbeh
#   bash scripts/run_baselines.sh all         # hotpot + gsm8k + openbookqa
#   bash scripts/run_baselines.sh all_benchmarks  # 上述 + 三个推理基准
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/run_baselines.sh gsm8k

set -e
cd "$(dirname "$0")/.."

DS="${1:-}"
if [[ -z "$DS" ]]; then
  echo "用法: bash scripts/run_baselines.sh <hotpot|gsm8k|openbookqa|super_gpqa|mmlu_pro|bbeh|all|all_benchmarks>"
  echo "  all：hotpot + gsm8k + openbookqa（与历史一致）"
  echo "  all_benchmarks：上述三者 + SuperGPQA + MMLU-Pro + BBEH"
  echo "环境变量: GPUS (默认 0,1,2,3,4,5,6,7)  LORA  SAMPLES  MODES"
  exit 1
fi

LORA=${LORA:-"outputs/qwen3_sr_lora_v3_base"}
SAMPLES=${SAMPLES:-200}
MODES=${MODES:-"B0,B1,B2,B4"}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}

IFS=',' read -r -a GPU_ARR <<< "$(echo "$GPUS" | tr -d '[:space:]')"
GPU_CLEAN=()
for g in "${GPU_ARR[@]}"; do
  [[ -n "$g" ]] && GPU_CLEAN+=("$g")
done
GPU_ARR=("${GPU_CLEAN[@]}")
N=${#GPU_ARR[@]}

if [[ "$N" -eq 0 ]]; then
  echo "错误: GPUS 为空"
  exit 1
fi

GPU_CSV=$(IFS=,; echo "${GPU_ARR[*]}")

mkdir -p logs

run_one() {
  local name="$1"
  local logf="logs/baseline_${name}.log"
  echo "=========================================="
  echo "  Baseline: $name"
  echo "  LoRA: $LORA  |  samples: $SAMPLES  |  modes: $MODES"
  echo "  --gpus $GPU_CSV  (共 $N 张) → $logf"
  echo "=========================================="
  python -u src/student/eval_baselines.py \
    --dataset "$name" \
    --lora_dir "$LORA" \
    --max_samples "$SAMPLES" \
    --modes "$MODES" \
    --gpus "$GPU_CSV" \
    2>&1 | tee "$logf"
}

case "$DS" in
  hotpot|gsm8k|openbookqa|super_gpqa|mmlu_pro|bbeh)
    run_one "$DS"
    echo ""
    grep -E "^\s+(B[0-4]|={3,}|数据集)" "logs/baseline_${DS}.log" || true
    ;;
  all)
    for name in hotpot gsm8k openbookqa; do
      run_one "$name"
      echo ""
    done
    echo "=========================================="
    echo "  汇总（grep 表格行）"
    echo "=========================================="
    for name in hotpot gsm8k openbookqa; do
      echo "=== $name ==="
      grep -E "^\s+(B[0-4]|={3,}|数据集)" "logs/baseline_${name}.log" || true
      echo ""
    done
    ;;
  all_benchmarks)
    for name in hotpot gsm8k openbookqa super_gpqa mmlu_pro bbeh; do
      run_one "$name"
      echo ""
    done
    echo "=========================================="
    echo "  汇总（grep 表格行）"
    echo "=========================================="
    for name in hotpot gsm8k openbookqa super_gpqa mmlu_pro bbeh; do
      echo "=== $name ==="
      grep -E "^\s+(B[0-4]|={3,}|数据集)" "logs/baseline_${name}.log" || true
      echo ""
    done
    ;;
  *)
    echo "未知数据集: $DS"
    exit 1
    ;;
esac
