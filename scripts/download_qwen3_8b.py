"""
将 Qwen3-8B 模型下载到项目 model 目录。
国内环境会使用 HF_ENDPOINT（hf-mirror）加速。
运行: python scripts/download_qwen3_8b.py
"""
import os

# 使用国内镜像（若已设置 HF_ENDPOINT 则沿用）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3-8B"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "model" / "Qwen3-8B"


def main() -> None:
    LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} to {LOCAL_DIR}...")
    print("(Using HF_ENDPOINT:", os.environ.get("HF_ENDPOINT", "default"), ")")
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
    )
    print(f"Done. Model saved to: {path}")


if __name__ == "__main__":
    main()
