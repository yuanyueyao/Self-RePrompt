# Self-RePrompt

使 LLM 在内部自动生成「优化后的 prompt 段」再作答，无需外部 prompt 优化器，思路类似 self-RAG。

**做法**：用特殊 token 约定模型先输出 `<SRP_START>…<SRP_END>` 段（自重写 prompt），再基于该段生成最终回答。难点在于训练数据的构建。

**本仓库**：基于 Qwen3-8B + LoRA 的实现，包含 Teacher 生成三元组数据、Student LoRA 训练与多数据集评测脚本。

## 核心思路拆解

- **特殊 token 约定**：约定一组只在内部使用的 token，例如：
  - `<SRP_START>`：开始自重写 prompt 段
  - `<SRP_END>`：结束自重写 prompt 段
  - （可选）`<SR_ANSWER>`：开始正式回答
- **生成模式**：模型在看到用户原始输入后，先生成一段“优化后的内部 prompt”（位于 `<SRP_START>` 与 `<SRP_END>` 之间），然后再基于这段内部 prompt 继续生成最终回答。

概念上的生成格式（实际训练与推理使用 Qwen3 的 `chat_template`，即 `user` / `assistant` 角色 + 上述特殊 token）：

```text
<USER>
原始用户指令……

<ASSISTANT>
<SRP_START>
（模型为自己写的、更结构化、更清晰的 prompt：拆解任务、明确约束、罗列步骤）
<SRP_END>
（从这里开始是最终给用户看的回答）
……
```

## 推理阶段流程设计

1. **拼接输入**  
   - 输入 = 系统提示（可选） + 历史对话（可选） + 当前用户指令。  
   - 不需要外部“prompt 优化器”，仅依赖模型内部学到的 `<SRP_START>` 生成能力。

2. **阶段一：自重写 prompt**  
   - 在训练/推理时，引导模型先输出 `<SRP_START>`。  
   - 模型继续生成，直到输出 `<SRP_END>`；中间内容即为“优化后的内部 prompt 段”。

3. **阶段二：基于内部 prompt 回答**  
   - `<SRP_END>` 之后，模型继续自回归生成，输出正式回答内容。  
   - 推理时可以：
     - 直接一次性生成（不做任何裁剪，只是内部使用 `<SRP_START>` 段），或者
     - 在产品层面隐藏 `<SRP_START>…<SRP_END>` 部分，只把 `<SRP_END>` 之后的内容展示给用户。

4. **可选：强制结构化**  
   - 通过系统提示或微调，让 `<SRP_START>` 段遵循固定结构，例如：
     - 任务重述
     - 关键信息提取
     - 约束与边界条件
     - 计划的解题步骤

## 训练数据构建方案

难点在于如何构造“原始输入 → 优化 prompt 段 → 最终回答”的训练样本。可以分为以下几类数据来源：

### 1. 从现有高质量对话中自动挖掘

- **数据来源**：已有的问答日志、开源对话数据集等。
- **自动构造思路**：
  - 使用一个较强的 teacher 模型（如 GPT 系列或更大模型）来执行“prompt 优化”任务。  
  - 给 teacher 的指令示例：

    ```text
    你是一个 prompt 优化器。
    给定用户指令和一个理想回答，请你写出一段内部使用的、结构化的 prompt，
    用于指导另一个模型去生成类似的回答。

    要求：
    - 明确任务目标
    - 列出关键约束和边界条件
    - 指定解题步骤或思考框架
    ```

  - 对于每条（用户指令，理想回答）样本，让 teacher 输出一个“优化后的 prompt 段”。  
  - 得到三元组：  
    - 原始输入：`U`  
    - 优化 prompt 段：`P*`  
    - 理想回答：`A*`

### 2. 人工标注少量高质量样本

- 选取典型任务场景（代码、写作、推理、多步骤工具调用等），人工设计：
  - 用户原始指令 `U`
  - 高质量“内部 prompt 段” `P*`
  - 对应的理想回答 `A*`
- 这些样本数量可以较少，但要求质量高，用于给模型一个非常清晰的模式模板。

### 3. 合成/增强数据

- 在已有三元组基础上做简单增强，例如：
  - 对 `U` 做轻微改写（同义替换、格式变化）  
  - 对 `P*` 保持结构不变，仅做少量措辞变化  
  - 对 `A*` 做局部重写或补充细节  
- 保持整体语义关系不变的前提下，扩充训练集规模。

## 训练目标与格式

### 1. 文本格式设计

每条训练样本对应三元组 `(U, P*, A*)`：原始输入 `U`、优化后的内部 prompt `P*`、基于 `P*` 的理想回答 `A*`。

- **概念格式**：`<USER> U <ASSISTANT> <SRP_START> P* <SRP_END> A*`
- **本仓库实现**：使用 Qwen3 的 `apply_chat_template`，`user` 角色填 `U`，`assistant` 角色填 `<SRP_START> P* <SRP_END> A*`；可选 `--mask_user` 只对 assistant 部分算 loss。
- **数据文件**：JSONL 每行 `{"user": "U", "sr_prompt": "P*", "srp_answer": "A*"}`，可选字段 `quadrant`；训练默认过滤 `misleading` 与 `both_wrong`（`--filter_quadrant` 可改）。

Teacher 模型负责生成 `P*` 与 `A*`，得到三元组后训练 Student；Student 在推理时对用户输入先生成 `<SRP_START>…<SRP_END>` 段，再生成最终回答。

### 2. Loss 设计（简单版本）

- **最简单做法**：对 assistant 输出部分（即 `<SRP_START>…<SRP_END>…A*`）的所有 token 做交叉熵损失；可选 `--mask_user` 只监督该部分，不监督 user 输入。  
- 这样模型会自然学到：先输出 `<SRP_START>`，在中间生成优化 prompt，再输出 `<SRP_END>` 与最终回答。

### 3. Loss 设计（加权版本，选做）

- 为了让模型更重视 `P*` 的质量，可以考虑：
  - 对 `<SRP_START>` 与 `<SRP_END>` 之间的 token 施加更高权重；
  - 或者单独做一个“prompt 质量判别”辅助任务，但这会复杂一些。

## 推理与工程实现细节

1. **不改模型结构，仅改训练与使用规范**  
   - 整个方案可以在现有自回归 LLM 上实现，不需要改动架构，只需要：
     - 增加特殊 token；
     - 构造对应格式的数据；
     - 进行持续微调或指令微调。

2. **推理时的两种使用方式**  
   - **直接展示全部输出**：适合研究与调试，便于观察 `<SRP_START>` 段是否合理。  
   - **仅展示 `<SRP_END>` 之后的内容**：适合产品上线，内部 prompt 段只作为“思考/提示”，不暴露给终端用户。

3. **与其他技巧的兼容性**  
   - 该方法可以与：
     - self-RAG（自检索、自总结知识库）  
     - chain-of-thought（显式思维链）  
     - tool calling（工具调用计划）  
     组合使用：`<SRP_START>` 段可以负责“计划与拆解”，后续回答阶段再按计划一步步执行。

## 项目结构

- **`data/`**：原始数据与生成的三元组（未纳入 Git，需本地准备或通过脚本生成）。
- **`src/teacher/`**：用 Teacher 模型生成 `sr_prompt` 与 `srp_answer` 的脚本（如 `gen_srp_prompt_from_*.py`、`gen_srp_answer_from_*.py`），支持 HotpotQA、GSM8K、OpenBookQA。
- **`src/student/`**：Student 训练与评测。
  - `train_qwen3_sr_lora.py`：Qwen3-8B + LoRA 训练，支持多 JSONL 合并、按 `quadrant` 过滤。
  - `eval_lora_accuracy.py`：对比 base 与 LoRA 的准确率，支持 HotpotQA、GSM8K、OpenBookQA、MATH。
- **`scripts/`**：数据下载、训练 `train_v3.sh` / `train_v3_base.sh`、`run_eval_base_example.sh`、**`compare_lora_v3_instruct_vs_base.sh`**（tmux 里挂跑：对比 Instruct+v3 LoRA 与 Base+v3_base LoRA，见脚本头注释）等。

## 环境与依赖

- Python 3.10+，建议使用 conda 环境。
- 主要依赖：`torch`、`transformers`、`peft`、`datasets`。基座模型为 **Qwen3-8B**（需自行下载到 `model/Qwen3-8B` 或使用 HuggingFace 名称）。

### Qwen3-8B-Base（预训练底座，对照实验）

- **与 Instruct 版区别**：`Qwen/Qwen3-8B-Base` 为预训练权重，对话与指令跟随通常弱于 `Qwen/Qwen3-8B`；**在 Base 上训练的 LoRA 不能加载到 Instruct 上，反之亦然**。
- **下载**：`python scripts/download_qwen3_8b_base.py` → 保存到 `model/Qwen3-8B-Base`。
- **训练（与 v3 相同数据，仅换底座）**：`bash scripts/train_v3_base.sh` → 输出 `outputs/qwen3_sr_lora_v3_base/`。过滤规则与 `train_v3.sh` 相同：默认 **`misleading` 与 `both_wrong`**（可用 `FILTER_QUADRANT=...` 覆盖）。
- **评测**：显式指定基座与 adapter，例如：
  ```bash
  bash scripts/run_eval_base_example.sh gsm8k 200   # 默认 8 卡并行（`GPUS=0,1,...` 可调）
  # 或
  python -u src/student/eval_lora_accuracy.py --base_model model/Qwen3-8B-Base \
      --lora_dir outputs/qwen3_sr_lora_v3_base --dataset gsm8k --max_samples 200 \
      --gpus 0,1,2,3,4,5,6,7
  ```
- Base 仍使用 tokenizer 的 `chat_template`（与官方仓库一致）；若后续官方变更导致模板缺失，需在训练/评测脚本中单独适配。

## 快速开始

1. **数据**  
   - 原始数据放在 `data/raw/`（如通过 `scripts/save_math_datasets.py` 下载 MATH/GSM8K）。  
   - 运行 teacher 脚本生成 `data/srp_prompt_with_answer/*.jsonl`（每行含 `user`、`sr_prompt`、`srp_answer`、`quadrant` 等）。

2. **训练（v3：三数据集 + 默认过滤 misleading 与 both_wrong）**  
   ```bash
   bash scripts/train_v3.sh
   ```  
   或指定 GPU：`GPUS=0,1,2,3 bash scripts/train_v3.sh`。输出 adapter 在 `outputs/qwen3_sr_lora_v3/`。仅过滤 misleading 时：`FILTER_QUADRANT=misleading bash scripts/train_v3.sh`。

3. **评测**  
   ```bash
   python -u src/student/eval_lora_accuracy.py --dataset gsm8k --max_samples 200 --lora_dir outputs/qwen3_sr_lora_v3
   ```  
   `--dataset` 可选：`hotpot`、`gsm8k`、`openbookqa`、`math`。MATH 支持 `--math_subject`、`--math_level` 过滤。

## 最小可行实验（MVP）建议

1. 选取一个已经较强的开源模型（例如 7B～14B 级别），在少量高质量数据上做指令微调。  
2. 手工或用 teacher 模型构造 1k～10k 条三元组样本。  
3. 采用简单的统一交叉熵 loss，专注验证两个问题：
   - 模型是否稳定地产生 `<SRP_START>…<SRP_END>` 段；
   - 该段落是否带来优于直接回答的质量提升。  
4. 对比指标：答案准确率（自动评测）、多步骤/多约束任务稳健性、`<SRP_START>` 段可解释性。

在验证 MVP 有明显收益后，再考虑扩大数据规模与更复杂的 loss 设计。