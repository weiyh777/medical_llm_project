import json
import os
import argparse
from tqdm import tqdm

def augment_cot_data(input_file, output_file):
    print(f"Enhancing data from {input_file} to {output_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    augmented_data = []
    
    # CoT System Prompt Template
    cot_instruction = (
        "你是一名专业的医学专家。请回答以下医学问题。\n"
        "请务必遵循以下格式进行回答：\n"
        "1. 首先，在 <think> 标签中进行深度的医学推理，分析症状、鉴别诊断和治疗原则。\n"
        "2. 然后，在 <answer> 标签中给出最终的、结构化的专业建议。\n\n"
        "例如：<think>...推理过程...</think><answer>...最终建议...</answer>"
    )
    
    for item in tqdm(data, desc="Augmenting"):
        original_prompt = item.get('prompt', '')
        # 如果是旧格式 question/answer
        if 'question' in item:
            original_prompt = item['question']
        else:
            # 尝试从prompt中提取纯问题
            original_prompt = original_prompt.replace("请回答以下医学问题：\n", "")
            
        new_item = {
            "prompt": [
                {"role": "system", "content": cot_instruction},
                {"role": "user", "content": original_prompt}
            ],
            # 保留原始答案作为Ground Truth用于奖励计算
            "answer": item['answer'] if 'answer' in item else item.get('chosen', '')
        }
        augmented_data.append(new_item)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Saved {len(augmented_data)} items to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="./data/grpo/train.jsonl")
    parser.add_argument("--output_file", default="./data/grpo/train.jsonl") 
    args = parser.parse_args()
    
    augment_cot_data(args.input_file, args.output_file)
