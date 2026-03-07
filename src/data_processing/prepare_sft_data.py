# -*- coding: utf-8 -*-
"""SFT数据处理脚本"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

class SFTDataProcessor:
    """SFT数据处理器"""
    
    SYSTEM_PROMPT = "你是一个专业的医学助手，具备丰富的医学知识和临床经验。请根据用户的问题，提供准确、专业的医学信息。"
    
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
    
    def convert_to_sharegpt(self, instruction: str, input_text: str, output: str) -> Dict:
        user_msg = f"{instruction}\n\n{input_text}" if input_text else instruction
        return {
            "system": self.SYSTEM_PROMPT,
            "conversations": [
                {"from": "human", "value": user_msg},
                {"from": "gpt", "value": output}
            ]
        }
    
    def process_medical_qa(self, input_file: str) -> List[Dict]:
        print(f"处理医学问答数据: {input_file}")
        data = self.load_jsonl(input_file)
        
        processed = []
        for item in tqdm(data, desc="处理医学问答"):
            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            output = item.get("output", "").strip()
            
            if not instruction or not output or len(output) < 10:
                continue
            
            processed.append(self.convert_to_sharegpt(instruction, input_text, output))
        
        print(f"处理完成，共 {len(processed)} 条样本")
        return processed
    
    def create_ceval_sft_data(self, ceval_dir: str) -> List[Dict]:
        import pandas as pd
        
        print(f"从C-Eval创建SFT数据: {ceval_dir}")
        subjects = ["basic_medicine", "clinical_medicine", "physician"]
        processed = []
        
        for subject in subjects:
            subject_dir = Path(ceval_dir) / subject
            for split in ["dev", "val"]:
                parquet_file = subject_dir / f"{split}-00000-of-00001.parquet"
                if not parquet_file.exists():
                    continue
                
                df = pd.read_parquet(parquet_file)
                for _, row in df.iterrows():
                    question = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
                    explanation = row.get("explanation", "")
                    answer = row["answer"]
                    
                    if explanation:
                        response = f"让我分析这道题目。\n\n{explanation}\n\n答案是：{answer}"
                    else:
                        response = f"根据医学知识分析，答案是：{answer}"
                    
                    processed.append(self.convert_to_sharegpt(question, "", response))
        
        print(f"C-Eval SFT数据: {len(processed)} 条")
        return processed
    
    def process_all(self, ceval_dir: Optional[str] = None):
        all_data = []
        
        zh_train = self.input_dir / "train_zh_0.json"
        if zh_train.exists():
            all_data.extend(self.process_medical_qa(str(zh_train)))
        
        en_train = self.input_dir / "train_en_1.json"
        if en_train.exists():
            all_data.extend(self.process_medical_qa(str(en_train)))
        
        if ceval_dir and Path(ceval_dir).exists():
            all_data.extend(self.create_ceval_sft_data(ceval_dir))
        
        import random
        random.seed(42)
        random.shuffle(all_data)
        
        split_idx = int(len(all_data) * 0.98)
        train_data = all_data[:split_idx]
        valid_data = all_data[split_idx:]
        
        self.save_jsonl(train_data, str(self.output_dir / "train.jsonl"))
        self.save_jsonl(valid_data, str(self.output_dir / "valid.jsonl"))
        
        print(f"SFT数据处理完成:")
        print(f"  - 训练集: {len(train_data)} 条")
        print(f"  - 验证集: {len(valid_data)} 条")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ceval_dir", type=str, default=None)
    parser.add_argument("--output_format", type=str, default="sharegpt")
    args = parser.parse_args()
    
    processor = SFTDataProcessor(args.input_dir, args.output_dir)
    processor.process_all(args.ceval_dir)


if __name__ == "__main__":
    main()
