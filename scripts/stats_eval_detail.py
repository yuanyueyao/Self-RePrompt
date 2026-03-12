"""
统计 eval 详情文件中的正确率、错误率、误导率、修正率。
运行: python scripts/stats_eval_detail.py [eval_detail.jsonl]
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 eval 详情")
    parser.add_argument(
        "file",
        type=str,
        default="eval/hotpot_train_qa_2000_reprompt_v3_eval_detail.jsonl",
        nargs="?",
        help="eval detail JSONL 文件路径",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"文件不存在: {path}")
        return

    with path.open("r", encoding="utf-8") as f:
        data = f.read().strip()
    # 支持 JSON 格式（含 meta + records）或 JSONL 格式
    meta = None
    try:
        obj = json.loads(data)
        if isinstance(obj, dict) and "records" in obj:
            rows = obj["records"]
            meta = obj.get("meta")
        else:
            rows = []
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in data.split("\n") if line.strip()]

    total = len(rows)
    if meta:
        direct_correct = meta["direct_correct"]
        sr_correct = meta["sr_correct"]
        both_correct = meta["both_correct"]
        both_wrong = meta["both_wrong"]
        misleading = meta["misleading"]
        corrected = meta["corrected"]
        direct_acc = meta["direct_accuracy"]
        sr_acc = meta["sr_accuracy"]
    else:
        direct_correct = sum(1 for r in rows if r.get("ok_direct"))
        sr_correct = sum(1 for r in rows if r.get("ok_sr"))
        both_correct = sum(1 for r in rows if r.get("ok_direct") and r.get("ok_sr"))
        both_wrong = sum(1 for r in rows if not r.get("ok_direct") and not r.get("ok_sr"))
        misleading = sum(1 for r in rows if r.get("ok_direct") and not r.get("ok_sr"))
        corrected = sum(1 for r in rows if not r.get("ok_direct") and r.get("ok_sr"))
        direct_acc = direct_correct / total if total else 0
        sr_acc = sr_correct / total if total else 0

    print("\n" + "=" * 50)
    print("Eval 统计结果")
    print("=" * 50)
    print(f" 总样本数: {total}")
    print()
    print("--- 各方法正确率/错误率 ---")
    print(f" 仅 user (direct):")
    print(f"   正确: {direct_correct}  准确率: {direct_acc:.3f}  错误率: {1-direct_acc:.3f}")
    print(f" user+sr_prompt:")
    print(f"   正确: {sr_correct}  准确率: {sr_acc:.3f}  错误率: {1-sr_acc:.3f}")
    print()
    print("--- sr_prompt 相对基线的效果 ---")
    print(f" 误导率 (本来对 → 加了 sr 后错): {misleading} / {direct_correct} ({misleading/direct_correct:.3f})" if direct_correct else " 误导率: N/A")
    print(f" 修正率 (本来错 → 加了 sr 后对): {corrected} / {total-direct_correct} ({corrected/(total-direct_correct):.3f})" if total-direct_correct else " 修正率: N/A")
    print()
    print("--- 四象限 ---")
    print(f" 都对:  {both_correct} ({both_correct/total:.1%})" if total else " 都对: N/A")
    print(f" 都错:  {both_wrong} ({both_wrong/total:.1%})" if total else " 都错: N/A")
    print(f" 误导:  {misleading} ({misleading/total:.1%}) (direct✓ sr✗)" if total else " 误导: N/A")
    print(f" 修正:  {corrected} ({corrected/total:.1%}) (direct✗ sr✓)" if total else " 修正: N/A")
    print("=" * 50)


if __name__ == "__main__":
    main()
