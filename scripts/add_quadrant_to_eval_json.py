"""
为已有 eval JSON 文件的每条 record 添加 quadrant 字段。
运行: python scripts/add_quadrant_to_eval_json.py [eval.json]
"""
import argparse
import json
from pathlib import Path


def quadrant_label(ok_d: bool, ok_s: bool) -> str:
    if ok_d and ok_s:
        return "both_correct"
    if not ok_d and not ok_s:
        return "both_wrong"
    if ok_d and not ok_s:
        return "misleading"
    return "corrected"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, nargs="?", default="eval/hotpot_train_qa_2000_reprompt_eval_detail.json")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"文件不存在: {path}")
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "records" not in data:
        print("文件中无 records 字段")
        return

    for rec in data["records"]:
        rec["quadrant"] = quadrant_label(rec.get("ok_direct"), rec.get("ok_sr"))

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已为 {len(data['records'])} 条 record 添加 quadrant 字段: {path}")


if __name__ == "__main__":
    main()
