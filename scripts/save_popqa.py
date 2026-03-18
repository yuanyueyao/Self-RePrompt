"""
从 HuggingFace 下载 PopQA 数据集，保存到 data/raw/popqa.json。

数据来源：akariasai/PopQA
    https://huggingface.co/datasets/akariasai/PopQA

PopQA 是实体中心的开放域问答数据集，适合评估/训练检索与事实记忆能力。

用法：
    python scripts/save_popqa.py
"""

import os

# 若未显式设置 HF_ENDPOINT，则默认使用 hf-mirror 镜像（国内可访问）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
from pathlib import Path

from datasets import load_dataset


def to_serializable(obj):
    """将 numpy 等类型转为 JSON 可序列化格式（与 save_math_datasets.py 保持一致）。"""
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


def save_popqa(out_dir: Path) -> None:
    print("Loading akariasai/PopQA ...")
    ds = load_dataset("akariasai/PopQA")

    # 官方只有一个 split（默认 "train"），这里直接全部收集到一个 list
    rows = []
    for split_name, split_ds in ds.items():
        print(f"  split={split_name}, size={len(split_ds)}")
        for i in range(len(split_ds)):
            row = to_serializable(dict(split_ds[i]))
            # 规范化字段名：question -> user，answers -> answer_candidates
            row["user"] = row.get("question", "")
            # PopQA 中答案字段通常为 answer / answers，此处不做强假设，仅归一存储
            if "answer" in row:
                row["gold_answer"] = row["answer"]
            if "answers" in row and "gold_answer" not in row:
                row["gold_answer"] = row["answers"]
            rows.append(row)

    out_path = out_dir / "popqa.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved {len(rows)} rows to {out_path}\n")


def main() -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_popqa(out_dir)
    print("Done. PopQA saved to data/raw/popqa.json")


if __name__ == "__main__":
    main()

