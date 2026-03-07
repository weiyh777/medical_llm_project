#!/bin/bash
# C-Eval医学评测脚本

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "C-Eval 医学科目评测"
echo "=========================================="

# 选择模型
# if [ -d "./outputs/grpo_model" ]; then
#     MODEL_PATH="./outputs/grpo_model"
#     MODEL_TYPE="GRPO"
# elif [ -d "./outputs/ppo_model" ]; then
#     MODEL_PATH="./outputs/ppo_model"
#     MODEL_TYPE="PPO"
# elif [ -d "./outputs/sft_model" ]; then
#     MODEL_PATH="./outputs/sft_model"
#     MODEL_TYPE="SFT"
# else
#     MODEL_PATH="./model/Qwen2.5-1.5B-Instruct"
#     MODEL_TYPE="Baseline"
# fi
MODEL_PATH="./model/Qwen2.5-7B-Instruct"
MODEL_TYPE="Baseline"

if [ ! -z "$1" ]; then
    MODEL_PATH="$1"
    MODEL_TYPE="Custom"
fi

CEVAL_DIR="../ceval-exam"
OUTPUT_DIR="./logs/eval_$(date +%Y%m%d_%H%M%S)"

echo "评测模型: $MODEL_PATH ($MODEL_TYPE)"

python src/evaluation/ceval_evaluator.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$CEVAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_shots 0 \
    --cot \
    --torch_dtype bfloat16

echo "评测完成！结果保存在: $OUTPUT_DIR"

# python src/evaluation/ceval_evaluator.py --model_path /root/autodl-tmp/medical_llm_project/model/Qwen2.5-1.5B-Instruct --data_dir /root/autodl-tmp/ceval-exam --output_dir /root/autodl-tmp/medical_llm_project/outputs/eval_baseline --n_shots 5 --cot --torch_dtype bfloat16
