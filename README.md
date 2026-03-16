# 医学领域大模型微调项目

## 项目目标

使用医疗对话数据，通过 **PT (预训练) + SFT (监督微调) + PPO/GRPO (强化学习)** 的训练流程，为大模型赋予医学推理能力，最终在C-Eval医疗相关评测集（basic_medicine, clinical_medicine, physician）上取得更优结果。

## 项目架构

```
medical_llm_project/
├── README.md                           # 项目说明
├── requirements.txt                    # 依赖包
├── configs/                            # 配置文件
│   ├── base_config.yaml               # 基础配置
│   ├── pt_config.yaml                 # 预训练配置
│   ├── sft_config.yaml                # SFT配置
│   ├── rm_config.yaml                 # 奖励模型配置
│   ├── ppo_config.yaml                # PPO配置
│   └── grpo_config.yaml               # GRPO配置
├── data/                               # 数据目录
│   ├── pretrain/                      # 预训练数据
│   ├── sft/                           # SFT数据
│   ├── reward/                        # 奖励模型数据
│   ├── grpo/                          # GRPO数据
│   └── eval/                          # 评测数据
├── scripts/                            # 运行脚本
│   ├── prepare_data.sh                # 数据准备脚本
│   ├── run_pt.sh                      # 预训练脚本
│   ├── run_sft.sh                     # SFT脚本
│   ├── run_rm.sh                      # 奖励模型训练脚本
│   ├── run_ppo.sh                     # PPO训练脚本
│   ├── run_grpo.sh                    # GRPO训练脚本
│   └── run_eval.sh                    # 评测脚本
├── src/                                # 源代码
│   ├── data_processing/               # 数据处理
│   ├── training/                       # 训练代码
│   ├── evaluation/                     # 评测代码
│   └── utils/                          # 工具函数
├── outputs/                            # 模型输出
└── logs/                               # 日志文件
```

## 数据集说明

### 训练数据来源 (medical/)

| 阶段 | 数据类型 | 来源文件 | 数据量 |
|-----|---------|---------|-------|
| PT (预训练) | 医学文本 | medical_book_zh.json | ~40MB |
| PT (预训练) | 医学百科 | train_encyclopedia.json | ~591MB |
| SFT (微调) | 中文问答 | train_zh_0.json | ~1.3GB |
| SFT (微调) | 英文问答 | train_en_1.json | ~139MB |
| RM/PPO | 偏好对比 | reward/train.json | ~3MB |

### 评测数据 (ceval-exam/)

| 科目 | dev | val | test |
|-----|-----|-----|------|
| basic_medicine (基础医学) | 5 | 19 | 175 |
| clinical_medicine (临床医学) | 5 | 22 | 200 |
| physician (执业医师) | 5 | 49 | 443 |

## 训练流程

### 阶段1: 领域预训练 (PT)
- **目的**: 让模型学习医学领域知识
- **数据**: 医学教材、医学百科
- **方法**: Causal LM + LoRA

### 阶段2: 监督微调 (SFT)
- **目的**: 让模型学会医学问答对话
- **数据**: 医学问答对话数据
- **方法**: Instruction Tuning + LoRA

### 阶段3: 奖励模型训练 (RM) [可选]
- **目的**: 学习评判回答质量
- **数据**: 偏好对比数据
- **方法**: Reward Modeling

### 阶段4: 强化学习 (PPO/GRPO)
- **目的**: 通过强化学习优化回答质量
- **方法**: PPO 或 GRPO (推荐)

## 快速开始

### 1. 安装依赖
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. 准备数据
\`\`\`bash
bash scripts/prepare_data.sh
\`\`\`

### 3. 训练流程
\`\`\`bash
# 阶段1: 预训练
bash scripts/run_pt.sh

# 阶段2: 监督微调
bash scripts/run_sft.sh

# 阶段3: 奖励模型 (可选)
bash scripts/run_rm.sh

# 阶段4: 强化学习 (选择 PPO 或 GRPO)
bash scripts/run_grpo.sh   # 推荐
# 或
bash scripts/run_ppo.sh
\`\`\`

### 4. 评测
\`\`\`bash
bash scripts/run_eval.sh
\`\`\`

### 5. 一键完整流程
\`\`\`bash
bash scripts/run_full_pipeline.sh
\`\`\`

## 模型选择建议

| 基座模型 | 参数量 | 显存需求 | 推荐场景 |
|---------|-------|---------|---------|
| Qwen2.5-0.5B | 0.5B | ~4GB | 快速验证 |
| Qwen2.5-1.5B | 1.5B | ~8GB | 平衡性能 |
| Qwen2.5-7B | 7B | ~24GB | 高性能 |
| Llama-3.2-3B | 3B | ~16GB | 英文较好 |

## GRPO 训练框架详解

### 核心训练框架：TRL (Transformer Reinforcement Learning)

`grpo_training_new.py` 使用 **Hugging Face TRL** 作为 GRPO 强化学习训练的核心框架，具体通过以下组件实现：

```python
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
```

| TRL 组件 | 作用 |
|---------|------|
| `GRPOTrainer` | 核心训练器，实现 GRPO (Group Relative Policy Optimization) 算法，管理训练循环、生成采样和策略更新 |
| `GRPOConfig` | GRPO 训练超参数配置，包括 `beta`（KL 散度惩罚系数）、`num_generations`（每个 prompt 的采样数量）、`max_prompt_length`、`max_completion_length` 等 |
| `ModelConfig` | 模型配置，包含 LoRA 参数（`lora_r`、`lora_alpha`、`lora_dropout`、`lora_target_modules`）以及量化选项（`load_in_4bit`、`load_in_8bit`） |
| `TrlParser` | 参数解析器，统一解析 `ModelConfig`、`ScriptArguments` 和 `GRPOConfig` 三组参数 |

### 配套依赖框架

代码中涉及的完整框架栈如下：

#### 1. Hugging Face Transformers（模型与 Tokenizer 加载）
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_zero3_enabled
```
- `AutoModelForCausalLM` 加载因果语言模型作为策略模型（policy model）
- `AutoTokenizer` 加载对应 tokenizer
- `BitsAndBytesConfig` 配置 4-bit/8-bit 量化（NF4 格式，双重量化）

#### 2. PEFT（参数高效微调 / LoRA）
```python
from peft import LoraConfig, TaskType, get_peft_model
```
- 通过 LoRA 仅训练少量低秩矩阵，大幅减少显存占用
- 支持 QLoRA（量化 + LoRA），通过 `--qlora` 开关控制
- 可配置目标模块（`q_proj`、`k_proj`、`v_proj`、`o_proj` 等 attention/FFN 层）
- `find_all_linear_names()` 函数可自动发现模型中所有线性层用于 LoRA

#### 3. PyTorch（深度学习基础框架）
```python
import torch
```
- 默认使用 `bfloat16` 精度训练（`torch.bfloat16`）
- 通过 `torch.cuda.device_count()` 检测 GPU 数量
- 通过 `torch.cuda.get_device_properties()` 获取 GPU 显存，自动分配内存

#### 4. 分布式训练支持
- **PyTorch DDP (DistributedDataParallel)**：通过 `torchrun --nproc_per_node` 启动多卡训练，按 `WORLD_SIZE` 自动调整 `gradient_accumulation_steps`
- **DeepSpeed ZeRO**：通过 `is_deepspeed_zero3_enabled()` 检测是否启用了 ZeRO-3。当同时开启量化（4-bit/8-bit）与 ZeRO-3 时，代码会抛出 `ValueError` 终止训练；仅开启 QLoRA 而未启用量化时则输出警告。

#### 5. Hugging Face Datasets（数据加载）
```python
from datasets import load_dataset
```
- 支持从 HuggingFace Hub 加载数据集（`dataset_name`）
- 支持从本地 JSON 目录加载（`train_file_dir`）

### GRPO 奖励函数设计

`GRPOTrainer` 接收多个奖励函数列表，本项目配置了三个奖励函数：

```python
trainer = GRPOTrainer(
    reward_funcs=[
        medical_content_reward,   # 内容相似度奖励
        ppl_penalty_reward,       # 困惑度惩罚奖励
        format_reward             # 格式合规奖励
    ],
    ...
)
```

| 奖励函数 | 实现方式 | 作用 |
|---------|---------|------|
| `medical_content_reward` | `difflib.SequenceMatcher` 计算与标准答案的字符级相似度（Ratio） | 鼓励回答内容与参考答案相近 |
| `ppl_penalty_reward` | 用策略模型自身计算生成文本的 PPL，`reward = -min(log(max(PPL, 1.0)), 5.0)`，取值范围 [-5.0, 0] | 惩罚困惑度高（语言不流畅）的生成 |
| `format_reward` | 正则匹配 `<think>.*</think><answer>.*</answer>` 格式 | 强制模型遵循「先思考后作答」的输出格式，奖励值为 0 或 1 |

### 训练流程总结

```
torchrun (DDP 多卡)
    └── GRPOTrainer (TRL)
            ├── AutoModelForCausalLM (Transformers) + LoRA (PEFT)
            ├── 量化支持: BitsAndBytes (4-bit NF4 / 8-bit)
            ├── 分布式: DeepSpeed ZeRO-2 兼容
            └── 奖励函数: medical_content + ppl_penalty + format
```

## 技术细节

- 使用 **LoRA/QLoRA** 进行高效微调
- 支持 **DeepSpeed ZeRO-2/3** 分布式训练
- 使用 **Flash Attention 2** 加速
- 支持 **梯度检查点** 节省显存
