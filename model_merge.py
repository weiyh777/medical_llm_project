from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('加载基础模型...')
base_model_path = './model/Qwen2.5-7B-Instruct'
adapter_path = './outputs/grpo_model_new_7b'
output_path = './outputs/grpo_model_merged_7B'

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

print('加载LoRA adapter...')
model = PeftModel.from_pretrained(model, adapter_path)

print('合并adapter到基础模型...')
model = model.merge_and_unload()

print(f'保存合并后的模型到 {output_path}...')
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print('完成!')