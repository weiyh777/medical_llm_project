# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train R1 model with GRPO rl algo.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import re
from datasets import load_dataset
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from peft import LoraConfig, TaskType, get_peft_model

import difflib
import math

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

REWARD_TOKENIZER = None
REWARD_MODEL = None
REWARD_MAX_LENGTH = 1024


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with GRPO
    """
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing training files for local datasets."}
    )
    train_samples: Optional[int] = field(default=-1, metadata={"help": "Number of samples to train on, -1 for all"})
    subset_name: Optional[str] = field(default="main",
                                       metadata={"help": "Subset name, e.g., 'default', 'main'. default is 'default'"})
    dataset_splits: Optional[str] = field(default="train", metadata={"help": "Split name"})
    preprocessing_num_workers: Optional[int] = field(default=10,
                                                     metadata={"help": "Number of workers for preprocessing"})
    # QLoRA arguments
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})


def normalize_text(text):
    """Normalize text by removing extra whitespace, converting to lowercase."""
    if text is None:
        return ""
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text


def extract_answer(text):
    """Extract content between <answer> tags."""
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def medical_content_reward(completions, answer, **kwargs):
    """Reward function that measures character-level string similarity between the model's answer and the ground truth.

    Algorithm (difflib.SequenceMatcher.ratio):
      1. Extract text inside <answer>...</answer> tags from the generated completion.
         If no tags are found, use the full completion text as the prediction.
      2. Run difflib.SequenceMatcher(None, pred, ground_truth) to find all
         non-overlapping matching character blocks between the two strings using
         the Ratcliff/Obershelp algorithm (longest common subsequence search).
      3. Compute the similarity ratio:
             ratio = 2 * M / T
         where M = total number of matched (overlapping) characters across all
         matching blocks, and T = total character count of both strings combined.
      4. Return the ratio directly as the reward. Value range: [0.0, 1.0].
         0.0 means no characters in common; 1.0 means the strings are identical.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, answer):
        # Extract the assistant's answer from <answer> tags
        match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        pred = match.group(1).strip() if match else content.strip()

        # Compute character-level similarity ratio: 2*M / T
        matcher = difflib.SequenceMatcher(None, pred, sol)
        rewards.append(matcher.ratio())
    return rewards

def _build_prompt_and_full_text(tokenizer, prompt_obj, completion_text: str):
    """构造 prompt_text 与 full_text，用于只在生成部分上计算 NLL。"""
    completion_text = str(completion_text).strip()

    # prompt 可能是 chat message list，也可能是纯字符串
    if isinstance(prompt_obj, list):
        messages = prompt_obj
    else:
        messages = [{"role": "user", "content": str(prompt_obj)}]

    if getattr(tokenizer, "chat_template", None):
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": completion_text}],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_plain = "\n".join([f"{m.get('role', 'user')}：{m.get('content', '')}" for m in messages])
        prompt_text = f"{prompt_plain}\nassistant："
        full_text = f"{prompt_text}{completion_text}"

    return prompt_text, full_text


def _calc_completion_ppl(tokenizer, model, prompt_text: str, full_text: str, max_length: int = 1024) -> float:
    """按 eval_ppl_test.py 思路：mask 掉 prompt，仅计算 completion 部分 PPL。"""
    device = next(model.parameters()).device

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

    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    labels = input_ids.clone()

    prompt_len = min(prompt_enc["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100

    # 至少一个可评估 token
    if (labels != -100).sum().item() == 0:
        return 1.0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        nll = outputs.loss.item()
    if was_training:
        model.train()

    # 数值稳定
    nll = min(20.0, max(0.0, nll))
    return float(math.exp(nll))


def ppl_penalty_reward(completions, **kwargs):
    """
    困惑度惩罚：PPL 越高，reward 越低（负值更大）。
    计算对象是完整生成文本（含 think + answer）。
    """
    global REWARD_TOKENIZER, REWARD_MODEL, REWARD_MAX_LENGTH

    contents = [completion[0]["content"] for completion in completions]

    # 兼容不同字段名
    prompts = kwargs.get("prompts", None)
    if prompts is None:
        prompts = kwargs.get("prompt", [""] * len(contents))

    if REWARD_TOKENIZER is None or REWARD_MODEL is None:
        logger.warning("PPL reward skipped: REWARD_TOKENIZER/REWARD_MODEL not initialized.")
        return [0.0] * len(contents)

    rewards = []
    for prompt_obj, content in zip(prompts, contents):
        try:
            prompt_text, full_text = _build_prompt_and_full_text(REWARD_TOKENIZER, prompt_obj, content)
            ppl = _calc_completion_ppl(
                REWARD_TOKENIZER,
                REWARD_MODEL,
                prompt_text,
                full_text,
                max_length=REWARD_MAX_LENGTH,
            )
            # 惩罚高 PPL：reward <= 0
            penalty = min(math.log(max(ppl, 1.0)), 5.0)
            rewards.append(float(-penalty))
        except Exception as e:
            logger.warning(f"PPL reward failed for one sample: {e}")
            rewards.append(0.0)

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    logger.debug(f'format rewards: {rewards}')
    return rewards


# Updated specific medical prompt
SYSTEM_PROMPT = (
    "你是一名专业的医学专家。请回答以下医学问题。"
    "请务必遵循以下格式进行回答："
    "1. 首先，在 <think> 标签中进行深度的医学推理，分析症状、鉴别诊断和治疗原则。"
    "2. 然后，在 <answer> 标签中给出最终的、结构化的专业建议。"
    "例如：<think>...推理过程...</think><answer>...最终建议...</answer>"
)


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def grpo_train(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    global REWARD_TOKENIZER, REWARD_MODEL, REWARD_MAX_LENGTH
    # Add distributed training initialization
    is_main_process = training_args.local_rank in [-1, 0]

    # Only log on main process
    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    if script_args.train_file_dir and os.path.exists(script_args.train_file_dir):
        # Load from local directory
        dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")
    else:
        # Load from HuggingFace hub
        dataset = load_dataset(script_args.dataset_name, script_args.subset_name, split=script_args.dataset_splits)

    if script_args.train_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(script_args.train_samples))

    # Prepare dataset
    with training_args.main_process_first(desc="Dataset preparation"):
        column_names = dataset.column_names
        dataset = dataset.map(
            lambda x: {
                'prompt': x['prompt'] if isinstance(x.get('prompt'), list) else [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': x['question'] if 'question' in x else x['prompt']}
                ],
                'answer': x['answer']
            },
            num_proc=script_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Processing dataset" if is_main_process else None,
        )

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    if is_main_process:
        logger.info("*** Initializing model kwargs ***")

    # Model initialization
    torch_dtype = torch.bfloat16  # Default to bfloat16 for compatibility

    # Set up distributed training config
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    # Check for QLoRA compatibility
    if script_args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")

    # Check quantization settings
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")

    # Set up quantization config
    quantization_config = None
    if script_args.qlora and (model_args.load_in_4bit or model_args.load_in_8bit):
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif model_args.load_in_4bit or model_args.load_in_8bit:
        # Support quantization even without qlora flag
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
    )

    num_gpus = torch.cuda.device_count()
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        model_kwargs["device_map"] = device_map
        # Ensure gradient_accumulation_steps is at least 1 after division
        training_args.gradient_accumulation_steps = max(training_args.gradient_accumulation_steps // world_size, 1)
    elif num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            total_mem = gpu_props.total_memory
            # 预留20%内存给训练时的梯度、优化器状态等
            usable_mem = int(total_mem * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    if is_main_process:
        logger.info(f"Using {num_gpus} GPUs")
        logger.info(f"model_kwargs={model_kwargs}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    
    REWARD_TOKENIZER = tokenizer
    REWARD_MODEL = model
    REWARD_MAX_LENGTH = getattr(training_args, "max_prompt_length", 1024) + getattr(training_args, "max_completion_length", 512)

    if is_main_process and hasattr(model, 'hf_device_map'):
        logger.info(f"Model Device Map: {model.hf_device_map.items()}")
    elif is_main_process and num_gpus > 1:
        logger.info("Model Device Map:")
        for name, param in model.named_parameters():
            if hasattr(param, 'device'):
                logger.info(f"  {name}: {param.device}")
                break

    # Configure LoRA if enabled
    if model_args.use_peft:
        if is_main_process:
            logger.info("Fine-tuning method: LoRA(PEFT)")
        if training_args.gradient_checkpointing:
            logger.warning("Gradient checkpointing is enabled. It may cause issues with LoRA, setting it to False.")
            training_args.gradient_checkpointing = False
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        if target_modules == 'all' or (target_modules and 'all' in target_modules):
            target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)
        if is_main_process:
            logger.info(f"Peft target_modules: {target_modules}, lora rank: {model_args.lora_r}, ")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        # Fixed FP16 ValueError for quantized models
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        if is_main_process:
            logger.info("Fine-tuning method: Full parameters training")

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")

    # Initialize GRPO trainer with distributed training support
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            medical_content_reward,
            ppl_penalty_reward,
            format_reward
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
    )
    logger.info("*** GRPO Trainer initialized ***")
    logger.debug(f"Trainer: {trainer}")

    # Training
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if is_main_process:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
            f'{training_args.num_train_epochs} epochs ***'
        )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log and save metrics on main process
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("*** Training complete ***")
        logger.info("*** Save model ***")

    # Save model
    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

        # Create model card and save config
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["r1", "grpo"],
        }
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if is_main_process:
        logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
