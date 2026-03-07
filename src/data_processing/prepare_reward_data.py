# -*- coding: utf-8 -*-
"""奖励模型数据处理脚本"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


class RewardDataProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def save_jsonl(self, data: List[Dict], file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def process_preference_data(self, input_file: str) -> List[Dict]:
        print(f"处理偏好对比数据: {input_file}")
        data = self.load_jsonl(input_file)
        
        processed = []
        for item in tqdm(data, desc="处理偏好数据"):
            question = item.get("question", "").strip()
            chosen = item.get("response_chosen", "").strip()
            rejected = item.get("response_rejected", "").strip()
            
            if not question or not chosen or not rejected or chosen == rejected:
                continue
            
            processed.append({
                "prompt": question,
                "chosen": chosen,
                "rejected": rejected
            })
        
        print(f"处理完成，共 {len(processed)} 条样本")
        return processed
    
    def process_all(self, output_format: str = "rm"):
        all_data = []
        
        train_file = self.input_dir / "train.json"
        if train_file.exists():
            all_data.extend(self.process_preference_data(str(train_file)))
        
        import random
        random.seed(42)
        random.shuffle(all_data)
        
        split_idx = int(len(all_data) * 0.9)
        train_data = all_data[:split_idx]
        valid_data = all_data[split_idx:]
        
        self.save_jsonl(train_data, str(self.output_dir / "train.jsonl"))
        self.save_jsonl(valid_data, str(self.output_dir / "valid.jsonl"))
        
        print(f"奖励模型数据处理完成:")
        print(f"  - 训练集: {len(train_data)} 条")
        print(f"  - 验证集: {len(valid_data)} 条")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_format", type=str, default="rm")
    args = parser.parse_args()
    
    processor = RewardDataProcessor(args.input_dir, args.output_dir)
    processor.process_all(args.output_format)


if __name__ == "__main__":
    main()
