"""
下载 MuSiQue 多跳 QA 数据集到 data/raw/。
国内服务器：从 HuggingFace (bdsaglam/musique) 下载，支持 hf-mirror。
运行: python scripts/save_musique.py
"""
import os

# 使用国内镜像（若已设置 HF_ENDPOINT 则沿用）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path

from huggingface_hub import hf_hub_download

# HuggingFace 上的 MuSiQue 文件（bdsaglam/musique）
REPO_ID = "bdsaglam/musique"
FILENAME = "musique_ans_v1.0_train.jsonl"  # 与官方 musique_ans_train.jsonl 内容一致
OUTPUT_NAME = "musique_ans_train.jsonl"


def main() -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / OUTPUT_NAME

    if output_path.exists():
        print(f"Not downloading {output_path} as it's already available locally.")
        return

    print(f"Downloading {FILENAME} from {REPO_ID} (via HF mirror)...")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    # 重命名为 musique_ans_train.jsonl
    actual_path = Path(downloaded)
    if actual_path.name != OUTPUT_NAME:
        actual_path.rename(output_path)
    print(f"\nDone. Saved to {output_path}")


if __name__ == "__main__":
    main()
