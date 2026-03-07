# -*- coding: utf-8 -*-
"""预训练数据处理脚本"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


class PretrainDataProcessor:
    """预训练数据处理器"""
    
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
    
    def process_medical_book(self, input_file: str) -> List[Dict]:
        print(f"处理医学教材数据: {input_file}")
        data = self.load_jsonl(input_file)
        
        processed = []
        for item in tqdm(data, desc="处理医学教材"):
            if 'text' in item and len(item['text'].strip()) > 100:
                processed.append({'text': item['text'].strip()})
        
        print(f"处理完成，共 {len(processed)} 条样本")
        return processed
    
    def process_encyclopedia(self, input_file: str) -> List[Dict]:
        print(f"处理医学百科数据: {input_file}")
        data = self.load_jsonl(input_file)
        
        processed = []
        for item in tqdm(data, desc="处理医学百科"):
            if 'text' in item and len(item['text'].strip()) > 50:
                text = item['text'].strip().replace('\r\n', '\n').replace('\r', '\n')
                processed.append({'text': text})
        
        print(f"处理完成，共 {len(processed)} 条样本")
        return processed
    
    def process_all(self):
        all_data = []
        
        medical_book_path = self.input_dir / "medical_book_zh.json"
        if medical_book_path.exists():
            all_data.extend(self.process_medical_book(str(medical_book_path)))
        
        train_encyclopedia = self.input_dir / "train_encyclopedia.json"
        if train_encyclopedia.exists():
            all_data.extend(self.process_encyclopedia(str(train_encyclopedia)))
        
        import random
        random.seed(42)
        random.shuffle(all_data)
        
        split_idx = int(len(all_data) * 0.98)
        train_data = all_data[:split_idx]
        valid_data = all_data[split_idx:]
        
        self.save_jsonl(train_data, str(self.output_dir / "train.jsonl"))
        self.save_jsonl(valid_data, str(self.output_dir / "valid.jsonl"))
        
        print(f"预训练数据处理完成:")
        print(f"  - 训练集: {len(train_data)} 条")
        print(f"  - 验证集: {len(valid_data)} 条")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    processor = PretrainDataProcessor(args.input_dir, args.output_dir)
    processor.process_all()


if __name__ == "__main__":
    main()
