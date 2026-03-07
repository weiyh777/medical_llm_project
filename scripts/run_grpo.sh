#!/bin/bash
# GRPO训练脚本 (推荐的强化学习方法)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "阶段4: GRPO强化学习训练"
echo "=========================================="

# if [ -d "./outputs/sft_model" ]; then
#     BASE_MODEL="./outputs/sft_model_filtered"
# else
#     BASE_MODEL="./model/Qwen2.5-1.5B-Instruct"
# fi
BASE_MODEL="./outputs/sft_model_merged_7B"

TRAIN_FILE_DIR="./data/grpo"
OUTPUT_DIR="./outputs/grpo_model_new_7b"

export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1

if [ ! -f "$TRAIN_FILE_DIR/train.jsonl" ]; then
    echo "错误: 未找到GRPO数据，请先运行 prepare_data.sh"
    exit 1
fi

echo "基座模型: $BASE_MODEL"

torchrun --nproc_per_node $NUM_GPUS ../MedicalGPT/grpo_training_new.py \
    --model_name_or_path "$BASE_MODEL" \
    --train_file_dir "$TRAIN_FILE_DIR" \
    --train_samples -1 \
    --max_steps -1 \
    --num_train_epochs 1 \
    --save_steps 100 \
    --save_strategy steps \
    --save_total_limit 3 \
    --output_dir "$OUTPUT_DIR" \
    --bf16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --beta 0.001 \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --use_vllm False \
    --logging_steps 10 \
    --use_peft True \
    --qlora False \
    --load_in_4bit False \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --num_generations 4 \
    --gradient_accumulation_steps 2 \
    --max_prompt_length 2048 \
    --max_completion_length 512

echo "GRPO训练完成！模型保存在: $OUTPUT_DIR"
