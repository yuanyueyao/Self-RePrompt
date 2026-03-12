"""
从 HuggingFace 下载 GSM8K 和 MATH 数学数据集，保存到 data/raw/。
默认使用 hf-mirror 镜像（国内可访问）。运行: python scripts/save_math_datasets.py
"""
import os

# 使用国内镜像，若已设置 HF_ENDPOINT 则沿用
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
import re
from pathlib import Path

from datasets import load_dataset


def to_serializable(obj):
    """将 numpy 等类型转为 JSON 可序列化格式"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    return obj


def extract_gsm8k_answer(answer_str: str) -> str:
    """从 GSM8K answer 中抽取最终答案（#### 后的数字）"""
    if not answer_str:
        return ""
    match = re.search(r"####\s*(.+)", answer_str.strip())
    return match.group(1).strip() if match else ""


def extract_math_answer(solution_str: str) -> str:
    """从 MATH solution 中抽取最终答案（\\boxed{} 中的内容）"""
    if not solution_str:
        return ""
    match = re.search(r"\\boxed\{([^}]*)\}", solution_str)
    return match.group(1).strip() if match else ""


def save_gsm8k(out_dir: Path) -> None:
    print("Loading openai/gsm8k (main)...")
    ds = load_dataset("openai/gsm8k", "main")

    result = {}
    for split in ds.keys():
        rows = []
        for i in range(len(ds[split])):
            row = to_serializable(dict(ds[split][i]))
            # 统一字段名，便于后续 gen_reprompt：question -> user, 抽取 answer
            row["question"] = row.get("question", "")
            row["answer_full"] = row.get("answer", "")
            row["answer"] = extract_gsm8k_answer(row.get("answer", ""))
            rows.append(row)
        result[split] = rows
        print(f"  {split}: {len(rows)} rows")

    out_path = out_dir / "gsm8k.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved to {out_path}\n")


MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def save_math(out_dir: Path) -> None:
    print("Loading EleutherAI/hendrycks_math (all 7 configs)...")
    all_train = []
    all_test = []
    for config in MATH_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", config)
        n_train, n_test = 0, 0
        for split_name, split_ds in ds.items():
            for i in range(len(split_ds)):
                row = to_serializable(dict(split_ds[i]))
                row["question"] = row.get("problem", "")
                row["answer_full"] = row.get("solution", "")
                row["answer"] = extract_math_answer(row.get("solution", ""))
                row["subject"] = config
                if split_name == "train":
                    all_train.append(row)
                    n_train += 1
                else:
                    all_test.append(row)
                    n_test += 1
        print(f"  {config}: train {n_train}, test {n_test}")

    result = {"train": all_train, "test": all_test}
    out_path = out_dir / "hendrycks_math.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  total: train {len(all_train)}, test {len(all_test)}")
    print(f"  -> Saved to {out_path}\n")


def main() -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_gsm8k(out_dir)
    save_math(out_dir)

    print("Done. GSM8K + MATH saved to data/raw/")


if __name__ == "__main__":
    main()
