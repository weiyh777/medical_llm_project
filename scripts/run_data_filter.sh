#!/bin/bash
# SFT数据筛选脚本
# 使用Embedding模型对训练数据进行相似度筛选

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "SFT数据筛选 - 基于Embedding相似度"
echo "=========================================="

# 配置参数
EMBEDDING_MODEL="./model/Qwen3-Embedding-0.6B"
CEVAL_EXAM_DIR="../ceval-exam"
SFT_TRAIN_FILE="./data/sft/train.jsonl"
OUTPUT_FILE="./data/sft/train_filtered.jsonl"

# 筛选参数
TOP_K=5                      # 每条语料选择的相关题目数量
TOP_N=10000                  # 按相似度排序后保留的训练样本数
CONTAMINATION_THRESHOLD=0.8 # 去污染阈值: 相似度>=该值的题目将被剔除
BATCH_SIZE=64                # 向量化批次大小

# 缓存目录
CACHE_DIR="./cache/embeddings"

# GPU设置
export CUDA_VISIBLE_DEVICES=0

# 检查必要文件
if [ ! -d "$EMBEDDING_MODEL" ]; then
    echo "错误: 未找到Embedding模型: $EMBEDDING_MODEL"
    echo "请先下载模型: huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir $EMBEDDING_MODEL"
    exit 1
fi

if [ ! -f "$SFT_TRAIN_FILE" ]; then
    echo "错误: 未找到SFT训练数据: $SFT_TRAIN_FILE"
    echo "请先运行 prepare_data.sh 准备数据"
    exit 1
fi

if [ ! -d "$CEVAL_EXAM_DIR" ]; then
    echo "错误: 未找到ceval-exam目录: $CEVAL_EXAM_DIR"
    exit 1
fi

echo ""
echo "配置信息:"
echo "  Embedding模型: $EMBEDDING_MODEL"
echo "  目标数据目录: $CEVAL_EXAM_DIR"
echo "  SFT数据文件: $SFT_TRAIN_FILE"
echo "  输出文件: $OUTPUT_FILE"
echo "  Top-K: $TOP_K"
echo "  保留样本数(Top-N): $TOP_N"
echo "  去污染阈值: $CONTAMINATION_THRESHOLD"
echo ""

# 创建缓存目录
mkdir -p "$CACHE_DIR"

# 运行数据筛选脚本
echo "开始数据筛选..."
python src/data_filter.py \
    --embedding_model "$EMBEDDING_MODEL" \
    --ceval_exam_dir "$CEVAL_EXAM_DIR" \
    --sft_train_file "$SFT_TRAIN_FILE" \
    --output_file "$OUTPUT_FILE" \
    --top_k $TOP_K \
    --top_n $TOP_N \
    --contamination_threshold $CONTAMINATION_THRESHOLD \
    --batch_size $BATCH_SIZE \
    --cache_dir "$CACHE_DIR" \
    --device cuda

# 检查输出文件
if [ -f "$OUTPUT_FILE" ]; then
    TOTAL_LINES=$(wc -l < "$SFT_TRAIN_FILE")
    FILTERED_LINES=$(wc -l < "$OUTPUT_FILE")
    echo ""
    echo "=========================================="
    echo "数据筛选完成！"
    echo "=========================================="
    echo "原始数据量: $TOTAL_LINES"
    echo "筛选后数据量: $FILTERED_LINES"
    echo "筛选后的数据保存在: $OUTPUT_FILE"
    echo ""
    echo "接下来可以使用筛选后的数据进行SFT训练:"
    echo "  修改 run_sft.sh 中的 TRAIN_FILE_DIR 或直接使用 run_sft_filtered.sh"
else
    echo "错误: 数据筛选失败，未生成输出文件"
    exit 1
fi
