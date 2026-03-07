import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 路径配置
project_dir = "/root/autodl-tmp/medical_llm_project"
base_model_path = os.path.join(project_dir, "model/Qwen2.5-7B-Instruct")
adapter_path = os.path.join(project_dir, "outputs/grpo_model_new_7b")

print(f"Base model: {base_model_path}")
print(f"Adapter: {adapter_path}")

if not os.path.exists(base_model_path):
    print("Error: Base model path not found!")
    exit(1)
if not os.path.exists(adapter_path):
    print("Error: Adapter path not found!")
    exit(1)

print("Loading base model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 示例病例输入
case_input = """
患者信息：
性别：男
年龄：45岁
主诉：反复上腹突发性绞痛伴呕吐3小时。
现病史：患者3小时前饱餐后突然出现上腹部绞痛，疼痛向右肩背部放射，伴恶心、呕吐，呕吐物为胃内容物。无发热、黄疸。
既往史：有胆囊结石病史5年。
查体：T 36.8℃，P 90次/分，BP 120/80mmHg。神志清，痛苦面容。右上腹压痛(+)，Murphy征(+)，无反跳痛及肌紧张。

请分析该病例可能的诊断并给出进一步检查建议。
"""

messages = [
    {"role": "system", "content": "你是一名经验丰富的医生，请根据患者的病例描述提供专业的诊断分析和医疗建议。"},
    {"role": "user", "content": case_input}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("-" * 50)
print("Input Prompt:")
print(text)
print("-" * 50)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating response...")
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("-" * 50)
print("Model Response:")
print(response)
print("-" * 50)
