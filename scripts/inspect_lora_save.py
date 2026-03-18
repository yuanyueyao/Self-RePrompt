#!/usr/bin/env python3
"""
查看 save_pretrained 保存的 LoRA adapter 目录内容。

用法：python scripts/inspect_lora_save.py [outputs/qwen3_sr_lora_v3]
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


def tree(dir_path: Path, prefix: str = "", max_depth: int = 3, depth: int = 0) -> str:
    if depth > max_depth:
        return ""
    lines = []
    if not dir_path.is_dir():
        return str(dir_path)
    entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    for i, p in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(str(prefix) + connector + p.name)
        if p.is_dir() and depth < max_depth:
            ext = "    " if is_last else "│   "
            lines.append(tree(p, prefix + ext, max_depth, depth + 1))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="查看 LoRA save_pretrained 保存内容")
    p.add_argument("path", nargs="?", default="outputs/qwen3_sr_lora_v3")
    args = p.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"路径不存在: {root}")
        sys.exit(1)

    print("=" * 70)
    print(f"  LoRA 保存目录: {root.resolve()}")
    print("=" * 70)

    # 1. 文件树
    print("\n【1】目录结构")
    print("-" * 50)
    print(tree(root))

    # 2. adapter_config.json
    cfg_path = root / "adapter_config.json"
    if cfg_path.exists():
        print("\n【2】adapter_config.json")
        print("-" * 50)
        cfg = json.loads(cfg_path.read_text())
        for k, v in cfg.items():
            print(f"  {k}: {v}")
    else:
        print("\n【2】adapter_config.json 不存在")

    # 3. tokenizer_config.json
    tok_cfg = root / "tokenizer_config.json"
    if tok_cfg.exists():
        print("\n【3】tokenizer_config.json")
        print("-" * 50)
        tc = json.loads(tok_cfg.read_text())
        for k, v in tc.items():
            print(f"  {k}: {v}")
    else:
        print("\n【3】tokenizer_config.json 不存在")

    # 4. adapter_model.safetensors
    safet = root / "adapter_model.safetensors"
    if safet.exists():
        print("\n【4】adapter_model.safetensors")
        print("-" * 50)
        if safe_open:
            with safe_open(safet, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                print(f"  共 {len(keys)} 个 tensor")
                for k in keys:
                    t = f.get_tensor(k)
                    print(f"    {k}: shape={t.shape}, dtype={t.dtype}")
        else:
            print("  (需安装 safetensors: pip install safetensors)")
    else:
        print("\n【4】adapter_model.safetensors 不存在")

    # 5. 其他文件
    print("\n【5】其他文件")
    print("-" * 50)
    others = ["tokenizer.json", "chat_template.jinja", "README.md"]
    for name in others:
        p = root / name
        if p.exists():
            size = p.stat().st_size
            print(f"  {name}: {size:,} bytes")
        else:
            print(f"  {name}: 不存在")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
