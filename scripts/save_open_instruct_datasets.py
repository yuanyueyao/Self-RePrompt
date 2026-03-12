"""
从 HuggingFace 下载 Open-Instruct 相关指令遵循类数据集，保存到 data/raw/。
默认使用 hf-mirror 镜像（国内可访问）。运行: python scripts/save_open_instruct_datasets.py

包含: GPT-4 Alpaca, Stanford Alpaca, FLAN-V2, ShareGPT, OpenAssistant
注意: FLAN-V2 约 246 万条，下载和保存较慢，请确保磁盘空间充足。
"""
import os

# 使用国内镜像，若已设置 HF_ENDPOINT 则沿用
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
from pathlib import Path

from datasets import load_dataset


# 数据集 ID 与保存文件名映射
DATASETS = [
    ("vicgalle/alpaca-gpt4", None, "gpt4_alpaca.json"),
    ("tatsu-lab/alpaca", None, "stanford_alpaca.json"),
    ("philschmid/flanv2", None, "flan_v2.json"),
    ("anon8231489123/ShareGPT_Vicuna_unfiltered", None, "sharegpt.json"),
    ("OpenAssistant/oasst1", None, "openassistant.json"),
]


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


def save_dataset(repo_id: str, config: str | None, out_name: str, out_dir: Path) -> None:
    print(f"Loading {repo_id} (config={config})...")
    kwargs = {"name": config} if config else {}
    ds = load_dataset(repo_id, **kwargs)

    result = {}
    for split in ds.keys():
        rows = [to_serializable(ds[split][i]) for i in range(len(ds[split]))]
        result[split] = rows
        print(f"  {split}: {len(rows)} rows")

    out_path = out_dir / out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved to {out_path}\n")


def main() -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    for repo_id, config, out_name in DATASETS:
        try:
            save_dataset(repo_id, config, out_name, out_dir)
        except Exception as e:
            print(f"ERROR {repo_id}: {e}\n")
            raise

    print("All datasets saved to data/raw/")


if __name__ == "__main__":
    main()
