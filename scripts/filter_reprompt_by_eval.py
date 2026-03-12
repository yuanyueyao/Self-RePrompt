"""
根据评测结果过滤 reprompt 数据，排除 misleading 样本（direct 对、sr 错）。
仅保留 both_correct 和 corrected，用于训练时避免 sr_prompt 拖累的样本。

使用方式:
    python scripts/filter_reprompt_by_eval.py
        --reprompt data/reprompt_reason/gsm8k_train_reprompt.jsonl
        --eval eval/gsm8k_train_reprompt_eval_Qwen3-8B.json
        --output data/reprompt_reason/gsm8k_train_reprompt_filtered.jsonl
"""
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据评测 JSON 排除 misleading 样本，输出过滤后的 reprompt JSONL。"
    )
    parser.add_argument("--reprompt", type=str, required=True, help="输入 reprompt JSONL 路径")
    parser.add_argument("--eval", type=str, required=True, help="评测结果 JSON 路径（含 meta 和 records）")
    parser.add_argument("--output", type=str, required=True, help="输出过滤后 JSONL 路径")
    parser.add_argument(
        "--keep_unmatched",
        action="store_true",
        help="若某行在 eval 中无对应记录，保留该行（默认丢弃）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    reprompt_path = Path(args.reprompt)
    eval_path = Path(args.eval)
    output_path = Path(args.output)

    with eval_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    records = eval_data.get("records", [])
    # idx -> quadrant
    idx_to_quadrant = {r["idx"]: r["quadrant"] for r in records}

    kept = 0
    dropped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with reprompt_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            quadrant = idx_to_quadrant.get(idx)
            if quadrant is None:
                if args.keep_unmatched:
                    fout.write(line + "\n")
                    kept += 1
                else:
                    dropped += 1
                continue

            if quadrant == "misleading":
                dropped += 1
                continue

            fout.write(line + "\n")
            kept += 1

    meta = eval_data.get("meta", {})
    total = meta.get("total", 0)
    misleading = meta.get("misleading", 0)
    print(f"输入: {reprompt_path} (eval 共 {total} 条)")
    print(f"misleading 样本: {misleading} 条")
    print(f"保留: {kept} 条")
    print(f"丢弃: {dropped} 条")
    print(f"输出: {output_path}")


if __name__ == "__main__":
    main()
