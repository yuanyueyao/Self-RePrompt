"""
将旧的 eval detail JSONL 转为带 meta 的 JSON 格式。
运行: python scripts/convert_eval_jsonl_to_json.py [input.jsonl] [output.json]
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="?", default="eval/hotpot_train_qa_2000_reprompt_v3_eval_detail.jsonl")
    parser.add_argument("output", type=str, nargs="?", default=None)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_suffix(".json")

    if not inp.exists():
        print(f"文件不存在: {inp}")
        return

    rows = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    total = len(rows)
    direct_correct = sum(1 for r in rows if r.get("ok_direct"))
    sr_correct = sum(1 for r in rows if r.get("ok_sr"))
    both_correct = sum(1 for r in rows if r.get("ok_direct") and r.get("ok_sr"))
    both_wrong = sum(1 for r in rows if not r.get("ok_direct") and not r.get("ok_sr"))
    misleading = sum(1 for r in rows if r.get("ok_direct") and not r.get("ok_sr"))
    corrected = sum(1 for r in rows if not r.get("ok_direct") and r.get("ok_sr"))

    def quadrant_label(ok_d: bool, ok_s: bool) -> str:
        if ok_d and ok_s:
            return "both_correct"
        if not ok_d and not ok_s:
            return "both_wrong"
        if ok_d and not ok_s:
            return "misleading"
        return "corrected"

    meta = {
        "total": total,
        "direct_correct": direct_correct,
        "direct_accuracy": round(direct_correct / total, 4),
        "sr_correct": sr_correct,
        "sr_accuracy": round(sr_correct / total, 4),
        "both_correct": both_correct,
        "both_correct_pct": round(both_correct / total, 4),
        "both_wrong": both_wrong,
        "both_wrong_pct": round(both_wrong / total, 4),
        "misleading": misleading,
        "misleading_pct": round(misleading / total, 4),
        "corrected": corrected,
        "corrected_pct": round(corrected / total, 4),
    }

    for rec in rows:
        rec["quadrant"] = quadrant_label(rec.get("ok_direct"), rec.get("ok_sr"))

    output = {"meta": meta, "records": rows}
    with out.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"已转换: {inp} -> {out}")


if __name__ == "__main__":
    main()
