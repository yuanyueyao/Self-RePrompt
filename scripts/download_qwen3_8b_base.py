"""
将 Qwen3-8B-Base（预训练底座，非 Instruct）下载到项目 model 目录。
与 Qwen3-8B（对话版）权重不同，需单独训练 LoRA，不可混用已有 adapter。

运行: python scripts/download_qwen3_8b_base.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3-8B-Base"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "model" / "Qwen3-8B-Base"


def main() -> None:
    LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} to {LOCAL_DIR}...")
    print("(Using HF_ENDPOINT:", os.environ.get("HF_ENDPOINT", "default"), ")")
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
    )
    print(f"Done. Model saved to: {path}")
    print(
        "\n下一步: bash scripts/train_v3_base.sh\n"
        "评测时请同时使用 --base_model model/Qwen3-8B-Base "
        "与 --lora_dir outputs/qwen3_sr_lora_v3_base"
    )


if __name__ == "__main__":
    main()
