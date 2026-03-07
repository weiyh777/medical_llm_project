#!/bin/bash
# 数据准备脚本

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "医学大模型训练数据准备"
echo "=========================================="

# 数据路径
MEDICAL_PRETRAIN_DIR="../medical/pretrain"
MEDICAL_SFT_DIR="../medical/finetune"
MEDICAL_REWARD_DIR="../medical/reward"
CEVAL_DIR="../ceval-exam"

# 输出路径
PRETRAIN_OUTPUT="./data/pretrain"
SFT_OUTPUT="./data/sft"
REWARD_OUTPUT="./data/reward"
GRPO_OUTPUT="./data/grpo"
EVAL_OUTPUT="./data/eval"

# 1. 准备预训练数据
echo "[1/5] 准备预训练数据..."
python src/data_processing/prepare_pretrain_data.py \
    --input_dir "$MEDICAL_PRETRAIN_DIR" \
    --output_dir "$PRETRAIN_OUTPUT"

# 2. 准备SFT数据
echo "[2/5] 准备SFT数据..."
python src/data_processing/prepare_sft_data.py \
    --input_dir "$MEDICAL_SFT_DIR" \
    --output_dir "$SFT_OUTPUT" \
    --ceval_dir "$CEVAL_DIR" \
    --output_format sharegpt

# 3. 准备奖励模型数据
echo "[3/5] 准备奖励模型数据..."
python src/data_processing/prepare_reward_data.py \
    --input_dir "$MEDICAL_REWARD_DIR" \
    --output_dir "$REWARD_OUTPUT" \
    --output_format rm

# 4. 准备GRPO数据
echo "[4/5] 准备GRPO数据..."
python src/data_processing/prepare_grpo_data.py \
    --output_dir "$GRPO_OUTPUT" \
    --reward_dir "$MEDICAL_REWARD_DIR"

# 5. 准备评测数据
echo "[5/5] 准备评测数据..."
python src/data_processing/prepare_eval_data.py \
    --input_dir "$CEVAL_DIR" \
    --output_dir "$EVAL_OUTPUT" \
    --n_shots 5 \
    --cot

echo "=========================================="
echo "数据准备完成！"
echo "=========================================="
