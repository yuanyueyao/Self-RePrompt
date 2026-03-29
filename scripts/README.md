# `scripts/` 目录说明

按用途分类；**仓库根目录**下通过 `bash scripts/...` 或 `python scripts/...` 调用路径不变。

---

## 1. Student 主线（训练 / 评测）


| 文件                 | 说明                                                            |
| ------------------ | ------------------------------------------------------------- |
| `train_v3.sh`      | **唯一训练入口**：Qwen3-8B-Base + 三数据集 LoRA，`torchrun` 多卡。           |
| `run_baselines.sh` | **唯一评测入口**：调用 `src/student/eval_baselines.py`，B0–B4，多 GPU 分片。 |


---

## 2. 模型与检查点


| 文件                          | 说明                                            |
| --------------------------- | --------------------------------------------- |
| `download_qwen3_8b_base.py` | 下载 **Qwen3-8B-Base** → `model/Qwen3-8B-Base`。 |
| `inspect_lora_save.py`      | 查看 LoRA `save_pretrained` 目录结构与权重 key。        |
| `check_cuda_env.py`         | 检查 `nvidia-smi` / `cuInit` / `torch.cuda` 是否一致（排错 802 等）。 |


---

## 3. 原始数据抓取 / 抽样

`data/raw/` 产出按任务可归为两类：**Math**（`save_math_datasets.py` → `gsm8k.json`、`hendrycks_math.json`）与 **通用推理**（其余下列脚本：问答、知识、综合推理基准、指令数据等）。


| 文件                               | 类别        | 说明                                         |
| -------------------------------- | ----------- | ------------------------------------------ |
| `save_math_datasets.py`          | Math        | GSM8K、MATH → `data/raw/`。                  |
| `save_reasoning_benchmarks.py`   | 通用推理      | SuperGPQA、MMLU-Pro、BBEH → `data/raw/`。    |
| `save_openbookqa.py`             | 通用推理      | OpenBookQA → `data/raw/`。                  |
| `save_musique.py`                | 通用推理      | MuSiQue → `data/raw/`。                     |
| `save_open_instruct_datasets.py` | 通用推理      | Alpaca / FLAN-V2 / ShareGPT / OASST 等。      |
| `save_popqa.py`                  | 通用推理      | PopQA → `data/raw/`。                       |
| `create_data_samples.py`         | —           | 从 `data/raw/` 抽样到 `data/raw/sample/`，便于预览。 |


---

## 4. Teacher / 评测产物处理（JSON / JSONL）


| 文件                              | 说明                          |
| ------------------------------- | --------------------------- |
| `filter_reprompt_by_eval.py`    | 按 eval 结果过滤 reprompt 数据。    |
| `add_quadrant_to_eval_json.py`  | 为 eval JSON 补充 quadrant 字段。 |
| `convert_eval_jsonl_to_json.py` | eval JSONL ↔ JSON 格式转换。     |
| `stats_eval_detail.py`          | eval 明细统计。                  |


---

## 使用顺序（典型）

1. `download_qwen3_8b_base.py`（若本地无基座）
2. `save_*.py` / `create_data_samples.py` 按需准备 `data/raw/`
3. `src/teacher/` 生成三元组（见 `src/README.md`）
4. `train_v3.sh` → `run_baselines.sh`

