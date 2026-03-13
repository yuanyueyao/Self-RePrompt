"""
从 data/raw/ 中各数据文件抽取样本，保存到 data/raw/sample/。
便于快速打开查看，不加载完整大文件。
运行: python scripts/create_data_samples.py
"""
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
SAMPLE_DIR = RAW_DIR / "sample"
SAMPLE_SIZE = 100  # 每个文件/每个 split 抽取的行数


def sample_jsonl(src: Path, dst: Path, n: int = SAMPLE_SIZE) -> None:
    """JSONL: 取前 n 行"""
    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(line + "\n")
            count += 1
            if count >= n:
                break
    print(f"  {src.name}: {count} lines -> {dst.name}")


def sample_json_array(src: Path, dst: Path, n: int = SAMPLE_SIZE) -> None:
    """JSON 数组: 取前 n 条"""
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        sampled = data[:n]
        with dst.open("w", encoding="utf-8") as f:
            json.dump(sampled, f, ensure_ascii=False, indent=2)
        print(f"  {src.name}: {len(data)} -> {len(sampled)} items -> {dst.name}")
    else:
        raise ValueError(f"Expected list, got {type(data)}")


def sample_json_splits(src: Path, dst: Path, n: int = SAMPLE_SIZE) -> None:
    """JSON 含 train/validation/test: 每个 split 取前 n 条"""
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict with splits, got {type(data)}")
    sampled = {}
    for split, rows in data.items():
        if isinstance(rows, list):
            sampled[split] = rows[:n]
            print(f"  {src.name} [{split}]: {len(rows)} -> {len(sampled[split])} items")
        else:
            sampled[split] = rows
    with dst.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"  -> {dst.name}")


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print("Creating samples in data/raw/sample/\n")

    # 遍历 data/raw 中所有数据文件（排除 .cache、sample 目录）
    for f in sorted(RAW_DIR.iterdir()):
        if f.is_dir() or f.name.startswith("."):
            continue
        dst = SAMPLE_DIR / f.name
        try:
            if f.suffix == ".jsonl":
                sample_jsonl(f, dst)
            elif f.suffix == ".json":
                with f.open("r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    sample_json_array(f, dst)
                elif isinstance(data, dict) and any(
                    isinstance(v, list) for v in data.values()
                ):
                    sample_json_splits(f, dst)
                else:
                    print(f"  {f.name}: skip (unknown structure)")
            else:
                print(f"  {f.name}: skip (unsupported format)")
        except Exception as e:
            print(f"  {f.name}: ERROR {e}")

    # srp_prompt 的 jsonl
    for name in ["hotpot_train_qa_2000_reprompt.jsonl", "gsm8k_train_reprompt.jsonl"]:
        reprompt = Path("data/srp_prompt") / name
        if reprompt.exists():
            sample_jsonl(reprompt, SAMPLE_DIR / name)

    print("\nDone.")


if __name__ == "__main__":
    main()
