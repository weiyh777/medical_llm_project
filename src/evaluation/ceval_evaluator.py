# -*- coding: utf-8 -*-
"""C-Eval医学评测器"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CEvalMedicalEvaluator:
    MEDICAL_SUBJECTS = ["basic_medicine", "clinical_medicine", "physician"]
    CHOICES = ["A", "B", "C", "D"]
    
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None,
                 device: str = "cuda", torch_dtype: str = "bfloat16",
                 n_shots: int = 5, cot: bool = False):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = device
        self.n_shots = n_shots
        self.cot = cot
        
        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True)
        
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype_map.get(torch_dtype, torch.bfloat16),
            device_map="auto", trust_remote_code=True)
        self.model.eval()
        print("模型加载完成")
    
    def load_subject_data(self, data_dir: str, subject: str, split: str = "test") -> pd.DataFrame:
        parquet_path = Path(data_dir) / subject / f"{split}-00000-of-00001.parquet"
        return pd.read_parquet(parquet_path)
    
    def format_example(self, row: pd.Series, include_answer: bool = False) -> str:
        example = f"问题：{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n"
        if include_answer:
            example += f"答案：{row['answer']}\n"
        return example
    
    def build_prompt(self, subject: str, test_row: pd.Series, dev_df: Optional[pd.DataFrame] = None) -> str:
        subject_cn = {"basic_medicine": "基础医学", "clinical_medicine": "临床医学", "physician": "执业医师"}.get(subject, subject)
        prompt = f"以下是关于{subject_cn}的单项选择题，请选出正确答案。\n\n"
        
        if dev_df is not None and self.n_shots > 0:
            for i in range(min(self.n_shots, len(dev_df))):
                prompt += f"示例{i+1}:\n{self.format_example(dev_df.iloc[i], include_answer=True)}\n"
        
        prompt += f"请回答以下问题：\n{self.format_example(test_row, include_answer=False)}"
        prompt += "请先分析推理，然后给出答案：\n" if self.cot else "答案："
        return prompt
    
    def extract_answer(self, response: str) -> str:
        patterns = [r"答案[是为：:]\s*([A-D])", r"选择?\s*([A-D])", r"^([A-D])[\.。,，\s]", r"([A-D])"]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return ""
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = prompt
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    def evaluate_subject(self, data_dir: str, subject: str) -> Tuple[float, List[Dict]]:
        print(f"评测科目: {subject}")
        test_df = self.load_subject_data(data_dir, subject, "test")
        try:
            dev_df = self.load_subject_data(data_dir, subject, "dev") if self.n_shots > 0 else None
        except:
            dev_df = None
        
        results = []
        correct = 0
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=subject):
            prompt = self.build_prompt(subject, row, dev_df)
            response = self.generate_response(prompt)
            predicted = self.extract_answer(response)
            
            result = {"id": row.get("id", idx), "question": row["question"], "predicted": predicted}
            if "answer" in row:
                result["ground_truth"] = row["answer"]
                result["correct"] = predicted == row["answer"]
                if result["correct"]:
                    correct += 1
            results.append(result)
        
        accuracy = correct / len(test_df) if len(test_df) > 0 else 0.0
        print(f"{subject} 准确率: {accuracy:.4f} ({correct}/{len(test_df)})")
        return accuracy, results
    
    def evaluate_all(self, data_dir: str, output_dir: Optional[str] = None) -> Dict:
        all_results = {}
        overall_correct, overall_total = 0, 0
        
        for subject in self.MEDICAL_SUBJECTS:
            try:
                accuracy, results = self.evaluate_subject(data_dir, subject)
                all_results[subject] = {"accuracy": accuracy, "results": results}
                overall_correct += sum(1 for r in results if r.get("correct", False))
                overall_total += len([r for r in results if "ground_truth" in r])
            except Exception as e:
                print(f"评测 {subject} 失败: {e}")
                all_results[subject] = {"error": str(e)}
        
        all_results["overall"] = {"accuracy": overall_correct/overall_total if overall_total else 0,
                                   "correct": overall_correct, "total": overall_total}
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(output_dir) / "eval_results.json", "w", encoding="utf-8") as f:
                summary = {k: {kk: vv for kk, vv in v.items() if kk != "results"} if isinstance(v, dict) else v
                          for k, v in all_results.items()}
                json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.print_summary(all_results)
        return all_results
    
    def print_summary(self, results: Dict):
        print("\n" + "="*60)
        print("C-Eval 医学科目评测结果")
        print("="*60)
        for subject in self.MEDICAL_SUBJECTS:
            if subject in results and "accuracy" in results[subject]:
                print(f"{subject:25s}: {results[subject]['accuracy']*100:.2f}%")
        print("-"*60)
        if "overall" in results:
            print(f"{'Overall':25s}: {results['overall']['accuracy']*100:.2f}%")
        print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_shots", type=int, default=5)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    args = parser.parse_args()
    
    evaluator = CEvalMedicalEvaluator(args.model_path, n_shots=args.n_shots, cot=args.cot, torch_dtype=args.torch_dtype)
    evaluator.evaluate_all(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
