#!/bin/bash
# 预训练脚本 (PT阶段) - 使用项目内置的pretraining.py

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "阶段1: 领域预训练 (PT)"
echo "=========================================="

BASE_MODEL="./model/Qwen2.5-1.5B-Instruct"
TRAIN_FILE="./data/pretrain/train.jsonl"
OUTPUT_DIR="./outputs/pt_model_new"

export CUDA_VISIBLE_DEVICES=0

# 检查数据文件
if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: 未找到预训练数据 $TRAIN_FILE，请先运行 prepare_data.sh"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

python src/training/pretraining.py \
    --model_name_or_path "$BASE_MODEL" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --block_size 1024 \
    --max_train_samples 50000 \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --gradient_checkpointing \
    --bf16 \
    --seed 42

echo "预训练完成！模型保存在: $OUTPUT_DIR"
