#!/bin/bash
# 完整训练流程脚本

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "医学大模型完整训练流程"
echo "=========================================="
echo "流程: 数据准备 -> PT -> SFT -> GRPO -> Eval"
echo ""

read -p "是否开始训练? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "已取消"
    exit 0
fi

START_TIME=$(date +%s)

echo "步骤 1/5: 数据准备"
bash scripts/prepare_data.sh

echo "步骤 2/5: 预训练 (PT)"
bash scripts/run_pt.sh

echo "步骤 3/5: 监督微调 (SFT)"
bash scripts/run_sft.sh

echo "步骤 4/5: 强化学习 (GRPO)"
bash scripts/run_grpo.sh

echo "步骤 5/5: 评测"
bash scripts/run_eval.sh

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo "=========================================="
echo "完整训练流程完成！"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟"
echo "=========================================="
