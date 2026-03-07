import os
import glob
import json
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_ceval_questions(data_dir):
    questions = []
    # Only look at val set for extraction as per user request (or should it be dev+val? usually val is fine)
    # The user said "CEval 验证集" (Validation Set)
    pattern = os.path.join(data_dir, "**", "val-*.parquet")
    files = glob.glob(pattern, recursive=True)
    
    for f in files:
        try:
            df = pd.read_parquet(f)
            for _, row in df.iterrows():
                q_text = row['question']
                # Append options to context? Maybe just question is enough for keywords.
                # Let's include options to be safe
                options = []
                for char in ['A', 'B', 'C', 'D']:
                    if pd.notna(row.get(char)):
                        options.append(f"{char}. {row[char]}")
                full_text = q_text + "\n" + "\n".join(options)
                questions.append(full_text)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return questions

def extract_keywords(model, tokenizer, questions, batch_size=4):
    keywords_list = []
    
    # Simple batching
    for i in tqdm(range(0, len(questions), batch_size), desc="Extracting Keywords"):
        batch_qs = questions[i:i+batch_size]
        prompts = []
        for q in batch_qs:
            prompt = f"你是一个医学教育专家。请仔细阅读下面的医学考试题目，提取出该题目考核的核心核心医学知识点、专有名词或考点关键词。请只输出关键词，用中文逗号分隔，不要包含“本题考查”、“知识点”等无关词汇。\n\n题目：\n{q}\n\n关键词："
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.1
            )
        
        # Decode
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for res in decoded:
            # Split by comma and clean
            # Handle user input variations like "A, B" or "A，B"
            raw_kws = res.replace("，", ",").split(",")
            kws = [k.strip() for k in raw_kws if k.strip()]
            keywords_list.extend(kws)
            
    return list(set(keywords_list))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./medical_llm_project/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--ceval_dir", type=str, default="./ceval-exam")
    parser.add_argument("--output_file", type=str, default="./medical_llm_project/data/sft/knowledge_points.json")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    print(f"Loading CEval questions from {args.ceval_dir}...")
    questions = load_ceval_questions(args.ceval_dir)
    print(f"Found {len(questions)} questions.")
    
    print(f"Loading model from {args.model_path}...")
    # Use bfloat16 for efficiency
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    unique_keywords = extract_keywords(model, tokenizer, questions, args.batch_size)
    
    print(f"Extracted {len(unique_keywords)} unique keywords.")
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_keywords, f, ensure_ascii=False, indent=2)
    print(f"Saved keywords to {args.output_file}")

if __name__ == "__main__":
    main()
