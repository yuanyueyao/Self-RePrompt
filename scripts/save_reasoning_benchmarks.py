"""
从 HuggingFace 下载通用推理类基准：SuperGPQA、MMLU-Pro、BBEH，保存到 data/raw/。
默认使用 hf-mirror 镜像（国内可访问）。运行: python scripts/save_reasoning_benchmarks.py

- m-a-p/SuperGPQA（研究生级多学科问答）
- TIGER-Lab/MMLU-Pro（MMLU 加强版，10 选）
- BBEH/bbeh（BIG-Bench Extra Hard，DeepMind）
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
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


def dataset_to_split_dict(ds_dict) -> dict:
    result = {}
    for split in ds_dict.keys():
        rows = [to_serializable(dict(ds_dict[split][i])) for i in range(len(ds_dict[split]))]
        result[split] = rows
        print(f"  {split}: {len(rows)} rows")
    return result


def save_super_gpqa(out_dir: Path) -> None:
    print("Loading m-a-p/SuperGPQA...")
    ds = load_dataset("m-a-p/SuperGPQA")
    result = dataset_to_split_dict(ds)
    out_path = out_dir / "super_gpqa.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved to {out_path}\n")


def save_mmlu_pro(out_dir: Path) -> None:
    print("Loading TIGER-Lab/MMLU-Pro...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    result = dataset_to_split_dict(ds)
    out_path = out_dir / "mmlu_pro.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved to {out_path}\n")


def save_bbeh(out_dir: Path) -> None:
    print("Loading BBEH/bbeh (BIG-Bench Extra Hard)...")
    ds = load_dataset("BBEH/bbeh")
    result = dataset_to_split_dict(ds)
    out_path = out_dir / "bbeh.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved to {out_path}\n")


def main() -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_super_gpqa(out_dir)
    save_mmlu_pro(out_dir)
    save_bbeh(out_dir)

    print("Done. SuperGPQA, MMLU-Pro, BBEH saved to data/raw/")


if __name__ == "__main__":
    main()
