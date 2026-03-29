#!/usr/bin/env python3
"""检查本机 GPU 驱动、libcuda 与 PyTorch CUDA 是否一致可用。"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys


def main() -> None:
    print("=== nvidia-smi ===")
    smi = shutil.which("nvidia-smi")
    if not smi:
        print("未找到 nvidia-smi（可能无 NVIDIA 驱动或未在 GPU 节点上）。")
    else:
        r = subprocess.run([smi, "-L"], capture_output=True, text=True)
        print(r.stdout or r.stderr or "(无输出)")

    print("\n=== libcuda.cuInit(0) ===")
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        print(f"无法加载 libcuda.so.1: {e}")
        return
    cu.cuInit.restype = ctypes.c_int
    rc = int(cu.cuInit(0))
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
    names = {
        0: "CUDA_SUCCESS",
        802: "CUDA_ERROR_SYSTEM_NOT_READY（驱动/节点未就绪或 GPU 子系统异常）",
        999: "CUDA_ERROR_UNKNOWN",
    }
    print(f"返回码: {rc}  ({names.get(rc, '见 CUDA driver API 文档')})")
    if rc != 0:
        print(
            "\n建议：在同一 shell 先跑 nvidia-smi 确认可见 GPU；仍报 802 时联系管理员检查\n"
            "驱动/nvidia-fabricmanager/GPU reset，或换到分配了 GPU 的计算节点。"
        )
        return

    print("\n=== PyTorch ===")
    try:
        import torch
    except ImportError:
        print("未安装 torch")
        return
    print(f"torch {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device 0: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
    sys.exit(0)
