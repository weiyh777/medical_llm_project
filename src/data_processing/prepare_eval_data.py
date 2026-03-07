# -*- coding: utf-8 -*-
"""评测数据处理脚本"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm


class EvalDataProcessor:
    MEDICAL_SUBJECTS = {
        "basic_medicine": "基础医学",
        "clinical_medicine": "临床医学",
        "physician": "执业医师"
    }
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_jsonl(self, data: List[Dict], file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def process_subject(self, subject: str, split: str = "test") -> List[Dict]:
        subject_dir = self.input_dir / subject
        parquet_file = subject_dir / f"{split}-00000-of-00001.parquet"
        
        if not parquet_file.exists():
            return []
        
        df = pd.read_parquet(str(parquet_file))
        processed = []
        
        for _, row in df.iterrows():
            item = {
                "id": int(row.get("id", len(processed))),
                "question": row["question"],
                "A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"],
                "subject": subject,
                "subject_cn": self.MEDICAL_SUBJECTS.get(subject, subject)
            }
            if "answer" in row:
                item["answer"] = row["answer"]
            if "explanation" in row and pd.notna(row["explanation"]):
                item["explanation"] = row["explanation"]
            processed.append(item)
        
        return processed
    
    def process_all(self, n_shots: int = 5, cot: bool = False):
        for subject in self.MEDICAL_SUBJECTS.keys():
            print(f"处理科目: {subject}")
            
            for split in ["dev", "val", "test"]:
                data = self.process_subject(subject, split)
                if data:
                    output_file = self.output_dir / f"{subject}_{split}.jsonl"
                    self.save_jsonl(data, str(output_file))
                    print(f"  {split}: {len(data)} 条")
        
        print(f"评测数据处理完成，输出目录: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_shots", type=int, default=5)
    parser.add_argument("--cot", action="store_true")
    args = parser.parse_args()
    
    processor = EvalDataProcessor(args.input_dir, args.output_dir)
    processor.process_all(args.n_shots, args.cot)


if __name__ == "__main__":
    main()
