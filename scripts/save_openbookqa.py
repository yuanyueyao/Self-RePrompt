"""
从 HuggingFace 加载 allenai/openbookqa (additional 配置) 并保存到 data/raw/。
默认使用 hf-mirror 镜像（国内可访问）。运行: python scripts/save_openbookqa.py
"""
import os

# 使用国内镜像，若已设置 HF_ENDPOINT 则沿用
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    print("Loading allenai/openbookqa (additional)...")
    ds = load_dataset("allenai/openbookqa", "additional")

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存为 JSON，按 split 分键
    result = {split: list(ds[split]) for split in ds.keys()}
    out_path = out_dir / "openbookqa_additional.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved to {out_path}")
    for split, rows in result.items():
        print(f"  {split}: {len(rows)} rows")


if __name__ == "__main__":
    main()
