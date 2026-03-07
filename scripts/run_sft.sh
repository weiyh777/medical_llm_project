#!/bin/bash
# 监督微调脚本 (SFT阶段)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "阶段2: 监督微调 (SFT)"
echo "=========================================="

# if [ -d "./outputs/pt_model_new" ]; then
#     BASE_MODEL="./outputs/pt_model_merged"
# else
#     BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
# fi
BASE_MODEL="./model/Qwen2.5-7B-Instruct"

TRAIN_FILE_DIR="./data/sft"
OUTPUT_DIR="./outputs/sft_model_7B"

export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1

if [ ! -f "$TRAIN_FILE_DIR/train.jsonl" ]; then
    echo "错误: 未找到SFT数据，请先运行 prepare_data.sh"
    exit 1
fi

echo "基座模型: $BASE_MODEL"

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
