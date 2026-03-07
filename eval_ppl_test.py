import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_concatenated_json(path: str) -> List[Dict]:
    """支持普通 JSON 数组、JSONL、以及“多个 JSON 对象直接拼接”的文件格式。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    stripped = text.strip()
    if not stripped:
        return []

    # 1) 先尝试标准 JSON（对象/数组）
    try:
        data = json.loads(stripped)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # 2) 尝试 JSONL（逐行 JSON）
    records: List[Dict] = []
    jsonl_ok = True
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
            else:
                jsonl_ok = False
                break
        except json.JSONDecodeError:
            jsonl_ok = False
            break
    if jsonl_ok and records:
        return records

    # 3) 回退：raw_decode 循环解析拼接对象
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    records = []
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        if isinstance(obj, dict):
            records.append(obj)
        idx = end

    return records


def build_prompt(example: Dict) -> str:
    instruction = str(example.get("instruction", "")).strip()
    user_input = str(example.get("input", "")).strip()
    if user_input:
        return f"{instruction}\n{user_input}"
    return instruction


def build_text_pair(
    tokenizer,
    prompt: str,
    output: str,
    use_chat_template: bool,
) -> Tuple[str, str]:
    output = str(output).strip()

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = f"用户：{prompt}\n助手："
        full_text = f"{prompt_text}{output}"

    return prompt_text, full_text


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class PPLDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int,
        use_chat_template: bool,
    ):
        self.samples: List[EncodedExample] = []

        for ex in data:
            output = str(ex.get("output", "")).strip()
            if not output:
                continue

            prompt = build_prompt(ex)
            prompt_text, full_text = build_text_pair(
                tokenizer=tokenizer,
                prompt=prompt,
                output=output,
                use_chat_template=use_chat_template,
            )

            full_enc = tokenizer(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            prompt_enc = tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = full_enc["input_ids"][0]
            attention_mask = full_enc["attention_mask"][0]
            labels = input_ids.clone()

            prompt_len = prompt_enc["input_ids"].shape[1]
            prompt_len = min(prompt_len, labels.shape[0])
            labels[:prompt_len] = -100

            # 至少保留一个可评估 token
            if (labels != -100).sum().item() == 0:
                continue

            self.samples.append(
                EncodedExample(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[EncodedExample], pad_token_id: int):
    max_len = max(x.input_ids.shape[0] for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        pad_len = max_len - x.input_ids.shape[0]

        input_ids.append(
            torch.cat(
                [x.input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]
            )
        )
        attention_mask.append(
            torch.cat([x.attention_mask, torch.zeros((pad_len,), dtype=torch.long)])
        )
        labels.append(
            torch.cat([x.labels, torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "labels": torch.stack(labels, dim=0),
    }


def evaluate_ppl(args):
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16 if args.fp16 else None,
    )
    model.to(device)
    model.eval()

    data = parse_concatenated_json(args.data_path)
    if not data:
        raise ValueError(f"未解析到有效数据: {args.data_path}")

    dataset = PPLDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_chat_template=args.use_chat_template,
    )
    if len(dataset) == 0:
        raise ValueError("数据经过处理后没有可评估样本（可能都被截断或 output 为空）。")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # loss 是对有效 token 的平均 NLL
            loss = outputs.loss
            valid_tokens = (labels[:, 1:] != -100).sum().item()
            if valid_tokens == 0:
                continue

            total_nll += loss.item() * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("没有可用于计算困惑度的 token。")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)

    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Samples used: {len(dataset)}")
    print(f"Evaluated tokens: {total_tokens}")
    print(f"Average NLL: {avg_nll:.6f}")
    print(f"Perplexity (PPL): {ppl:.6f}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="在 test_zh_0.json 上计算模型困惑度")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="本地模型目录或 HF 模型名",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/autodl-tmp/medical/finetune/test_zh_0.json",
        help="测试数据路径",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fp16", action="store_true", help="启用 fp16 推理")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="加载需要 remote code 的模型",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="使用 tokenizer chat template 构造对话（建议指令模型开启）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_ppl(args)
