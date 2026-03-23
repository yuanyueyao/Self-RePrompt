#!/usr/bin/env bash
# =============================================================================
# 对比：Qwen3-8B + outputs/qwen3_sr_lora_v3
#       vs Qwen3-8B-Base + outputs/qwen3_sr_lora_v3_base
# 基座与 adapter 必须一一对应，不可混挂。
#
# tmux 里挂着跑（推荐）：
#   cd /data3/yyy/Self-RePrompt
#   tmux new -s cmp
#   bash scripts/compare_lora_v3_instruct_vs_base.sh
#   # Ctrl+B D  detach；之后: tmux attach -t cmp
#
# 可调环境变量：
#   CONDA_ENV=srp          # conda 环境名
#   SAMPLES=200            # 每个数据集采样条数
#   GPUS=0,1,2,3,4,5,6,7   # 多卡并行（传给 eval_lora_accuracy --gpus）
#   DATASETS="gsm8k hotpot openbookqa"   # 只跑部分集
#
# 日志：
#   logs/compare_instruct_vs_base_MASTER_<时间戳>.log  （整次运行总览）
#   logs/compare_instruct_v3_<dataset>.log
#   logs/compare_base_v3_<dataset>.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-srp}"
SAMPLES="${SAMPLES:-200}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
# shellcheck disable=SC2206
DATASETS=( ${DATASETS:-gsm8k hotpot openbookqa} )

mkdir -p logs
MASTER="logs/compare_instruct_vs_base_MASTER_$(date +%Y%m%d_%H%M%S).log"
export PYTHONUNBUFFERED=1

# 总日志：终端 + 文件（tmux attach 仍能看到实时输出）
exec > >(tee -a "$MASTER") 2>&1

echo "=============================================="
echo "  START $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "  cwd: $(pwd)"
echo "  CONDA_ENV=$CONDA_ENV  SAMPLES=$SAMPLES  GPUS=$GPUS"
echo "  DATASETS: ${DATASETS[*]}"
echo "  MASTER_LOG: $MASTER"
echo "=============================================="

if command -v conda &>/dev/null; then
  run_py() { conda run -n "$CONDA_ENV" --no-capture-output python -u "$@"; }
else
  run_py() { python -u "$@"; }
fi

run_one() {
  local tag="$1" base="$2" lora="$3" ds="$4"
  local logf="logs/compare_${tag}_${ds}.log"
  echo ""
  echo ">>> [$(date +%H:%M:%S)] $tag | dataset=$ds | n=$SAMPLES"
  echo "    base=$base  lora=$lora"
  echo "    tee → $logf"
  run_py src/student/eval_lora_accuracy.py \
    --base_model "$base" \
    --lora_dir "$lora" \
    --dataset "$ds" \
    --max_samples "$SAMPLES" \
    --gpus "$GPUS" 2>&1 | tee "$logf"
}

for ds in "${DATASETS[@]}"; do
  run_one "instruct_v3" "model/Qwen3-8B" "outputs/qwen3_sr_lora_v3" "$ds"
  run_one "base_v3" "model/Qwen3-8B-Base" "outputs/qwen3_sr_lora_v3_base" "$ds"
done

echo ""
echo "========== 汇总（各数据集）=========="
for ds in "${DATASETS[@]}"; do
  echo ""
  echo "--- $ds ---"
  echo "  [Instruct + qwen3_sr_lora_v3]"
  grep -E "Base 准确率|LoRA 准确率|净增益|SRP 格式触发率|四象限|both_correct|corrected|both_wrong|misleading" "logs/compare_instruct_v3_${ds}.log" 2>/dev/null | tail -16 || echo "  (无匹配，请打开 logs/compare_instruct_v3_${ds}.log)"
  echo "  [Base + qwen3_sr_lora_v3_base]"
  grep -E "Base 准确率|LoRA 准确率|净增益|SRP 格式触发率|四象限|both_correct|corrected|both_wrong|misleading" "logs/compare_base_v3_${ds}.log" 2>/dev/null | tail -16 || echo "  (无匹配，请打开 logs/compare_base_v3_${ds}.log)"
done

echo ""
echo "=============================================="
echo "  DONE $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "  总日志: $MASTER"
echo "=============================================="
