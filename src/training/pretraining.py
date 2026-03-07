#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的预训练脚本 - 兼容transformers 5.0+
使用LoRA进行领域预训练
"""

import os
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Domain pretraining with LoRA")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="Whether to trust remote code")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Torch dtype for model")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data file (jsonl format)")
    parser.add_argument("--validation_file", type=str, default=None,
                        help="Path to validation data file")
    parser.add_argument("--block_size", type=int, default=1024,
                        help="Block size for tokenization")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum number of training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Maximum number of evaluation samples")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4,
                        help="Number of preprocessing workers")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Whether to use LoRA")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Target modules for LoRA")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--num_train_epochs", type=float, default=0.5,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--bf16", action="store_true",
                        help="Whether to use bf16")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use fp16")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def get_torch_dtype(dtype_str):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def load_and_prepare_data(args, tokenizer):
    """加载并准备预训练数据"""
    logger.info(f"Loading training data from {args.train_file}")
    
    # 加载数据
    data_files = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file
    
    raw_datasets = load_dataset("json", data_files=data_files)
    
    # 限制样本数量
    if args.max_train_samples:
        raw_datasets["train"] = raw_datasets["train"].select(
            range(min(args.max_train_samples, len(raw_datasets["train"])))
        )
    if args.validation_file and args.max_eval_samples:
        raw_datasets["validation"] = raw_datasets["validation"].select(
            range(min(args.max_eval_samples, len(raw_datasets["validation"])))
        )
    
    logger.info(f"Training samples: {len(raw_datasets['train'])}")
    if "validation" in raw_datasets:
        logger.info(f"Validation samples: {len(raw_datasets['validation'])}")
    
    # Tokenize函数
    def tokenize_function(examples):
        # 假设数据格式为 {"text": "..."}
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.block_size,
            padding="max_length",
            return_attention_mask=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # 处理数据
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing",
    )
    
    return tokenized_datasets


def main():
    args = parse_args()
    
    logger.info("=" * 50)
    logger.info("Domain Pretraining with LoRA")
    logger.info("=" * 50)
    
    # 设置torch dtype
    torch_dtype = get_torch_dtype(args.torch_dtype)
    
    # 加载tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    logger.info(f"Loading model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )
    
    # 配置LoRA
    if args.use_lora:
        logger.info("Configuring LoRA...")
        target_modules = args.target_modules.split(",")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 启用梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    # 准备数据
    tokenized_datasets = load_and_prepare_data(args, tokenizer)
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # CLM, not MLM
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if args.validation_file else "no",
        eval_steps=args.eval_steps if args.validation_file else None,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # 保存模型
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logger.info("Training completed!")
    logger.info(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
