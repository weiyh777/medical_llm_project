#!/bin/bash
# 使用筛选后数据的监督微调脚本 (SFT阶段)
# 该脚本使用经过Embedding相似度筛选的高质量数据进行训练

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "阶段2: 监督微调 (SFT) - 使用筛选后的数据"
echo "=========================================="

# 检查是否已经完成数据筛选
FILTERED_TRAIN_FILE="./data/sft/train_filtered.jsonl"
if [ ! -f "$FILTERED_TRAIN_FILE" ]; then
    echo "未找到筛选后的数据文件: $FILTERED_TRAIN_FILE"
    echo "正在运行数据筛选..."
    bash scripts/run_data_filter.sh
fi

# 基座模型配置
BASE_MODEL="./model/Qwen2.5-7B-Instruct"

# 创建临时数据目录，包含筛选后的训练数据
TRAIN_FILE_DIR="./data/sft_filtered_7b"
mkdir -p "$TRAIN_FILE_DIR"

# 复制筛选后的训练数据
cp "$FILTERED_TRAIN_FILE" "$TRAIN_FILE_DIR/train.jsonl"

# 复制验证集（使用原始验证集或也可以筛选）
if [ -f "./data/sft/valid.jsonl" ]; then
    cp "./data/sft/valid.jsonl" "$TRAIN_FILE_DIR/valid.jsonl"
fi

OUTPUT_DIR="./outputs/sft_model_filtered_7b"

export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1

echo "基座模型: $BASE_MODEL"
echo "训练数据: $TRAIN_FILE_DIR/train.jsonl"
echo "数据量: $(wc -l < $TRAIN_FILE_DIR/train.jsonl) 条"

torchrun --nproc_per_node $NUM_GPUS ../MedicalGPT/supervised_finetuning.py \
    --model_name_or_path "$BASE_MODEL" \
    --train_file_dir "$TRAIN_FILE_DIR" \
    --validation_file_dir "$TRAIN_FILE_DIR" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples 100000 \
    --max_eval_samples 100 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_steps 10 \
    --eval_steps 500 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --output_dir "$OUTPUT_DIR" \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir ./cache \
    --flash_attn True

echo "SFT训练完成！模型保存在: $OUTPUT_DIR"
