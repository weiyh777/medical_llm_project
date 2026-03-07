# -*- coding: utf-8 -*-
"""GRPO数据处理脚本 - 使用医学偏好数据"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import random


class GRPODataProcessor:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """加载JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def save_jsonl(self, data: List[Dict], file_path: str):
        """保存为JSONL格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def process_reward_data(self, reward_dir: str) -> List[Dict]:
        """
        处理医学偏好数据 (chosen/rejected格式)
        转换为GRPO训练格式 (question/answer)
        """
        print(f"处理医学偏好数据: {reward_dir}")
        processed = []
        
        train_file = Path(reward_dir) / "train.json"
        if not train_file.exists():
            print(f"警告: 未找到 {train_file}")
            return processed
        
        data = self.load_jsonl(str(train_file))
        
        for item in tqdm(data, desc="处理偏好数据"):
            question = item.get("question", "")
            chosen = item.get("response_chosen", "")
            
            if question and chosen:
                # MedicalGPT GRPO格式: question + answer
                # 使用chosen作为正确答案
                processed.append({
                    "question": question,
                    "answer": chosen
                })
        
        print(f"偏好数据处理完成: {len(processed)} 条")
        return processed
    
    def process_all(self, reward_dir: str, train_ratio: float = 0.95):
        """处理所有数据并划分训练/验证集"""
        all_data = []
        
        # 处理偏好数据
        if reward_dir and Path(reward_dir).exists():
            all_data.extend(self.process_reward_data(reward_dir))
        
        if not all_data:
            print("错误: 没有找到可用的GRPO训练数据")
            return
        
        # 打乱数据
        random.seed(42)
        random.shuffle(all_data)
        
        # 划分训练集和验证集
        split_idx = int(len(all_data) * train_ratio)
        train_data = all_data[:split_idx]
        valid_data = all_data[split_idx:]
        
        # 保存
        self.save_jsonl(train_data, str(self.output_dir / "train.jsonl"))
        self.save_jsonl(valid_data, str(self.output_dir / "valid.jsonl"))
        
        print(f"GRPO数据处理完成:")
        print(f"  训练集: {len(train_data)} 条")
        print(f"  验证集: {len(valid_data)} 条")


def main():
    parser = argparse.ArgumentParser(description="准备GRPO训练数据")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--reward_dir", type=str, required=True,
                        help="医学偏好数据目录 (包含chosen/rejected)")
    args = parser.parse_args()
    
    processor = GRPODataProcessor(args.output_dir)
    processor.process_all(args.reward_dir)


if __name__ == "__main__":
    main()
