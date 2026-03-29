# `src/` 源码布局

---

## `src/student/` — Student（LoRA 微调与 baseline 评测）

| 文件 | 说明 |
|------|------|
| `train_qwen3_sr_lora.py` | LoRA 训练实现；由 **`scripts/train_v3.sh`** 调用。 |
| `eval_baselines.py` | B0–B4 评测实现；由 **`scripts/run_baselines.sh`** 调用。 |
| `test_train_pipeline.py` | 可选：tokenizer / `mask_user` / 小规模加载自检，**非**训练入口。 |

---

## `src/teacher/` — Teacher（数据构造与 API 评测）

| 类别 | 文件 |
|------|------|
| 生成 `sr_prompt` | `gen_srp_prompt_from_hotpot.py`、`gen_srp_prompt_from_gsm8k.py`、`gen_srp_prompt_from_openbookqa.py` |
| 生成 `srp_answer` | `gen_srp_answer_from_hotpot.py`、`gen_srp_answer_from_gsm8k.py`、`gen_srp_answer_from_openbookqa.py` |
| API 辅助 | `gen_sr_answer_by_api.py`、`filter_correct_by_api.py` |
| Teacher 上 direct vs sr_prompt 效果 / 四象限 | `eval_hotpot_sr_effect_question.py`、`eval_gsm8k_sr_effect_question.py` |

与 Student 评测（`run_baselines.sh`）**不是**同一套脚本：Teacher 侧面向云端/强模型与训练数据标注。
