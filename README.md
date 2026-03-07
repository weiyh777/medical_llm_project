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

## 技术细节

- 使用 **LoRA/QLoRA** 进行高效微调
- 支持 **DeepSpeed ZeRO-2/3** 分布式训练
- 使用 **Flash Attention 2** 加速
- 支持 **梯度检查点** 节省显存
