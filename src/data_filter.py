#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据筛选脚本：使用Embedding模型对SFT数据进行筛选
基于与目标分布(ceval-exam医学数据)的相似度选择最相关的训练数据
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer


def load_ceval_exam_data(ceval_exam_dir: str) -> List[str]:
    """
    加载ceval-exam目录下的所有医学考试数据
    将题目和选项拼接成完整的文本用于向量化
    """
    texts = []
    ceval_path = Path(ceval_exam_dir)
    
    # 遍历所有子目录
    for subject_dir in ceval_path.iterdir():
        if not subject_dir.is_dir():
            continue
            
        # 读取所有parquet文件
        for parquet_file in subject_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                
                # 将每道题目拼接成完整文本
                for _, row in df.iterrows():
                    # 构建题目文本：题目 + 选项
                    question = str(row.get('question', ''))
                    options = []
                    for opt in ['A', 'B', 'C', 'D']:
                        if opt in row and pd.notna(row[opt]):
                            options.append(f"{opt}. {row[opt]}")
                    
                    # 拼接题目和选项
                    full_text = question
                    if options:
                        full_text += "\n" + "\n".join(options)
                    
                    if full_text.strip():
                        texts.append(full_text)
                        
            except Exception as e:
                print(f"读取文件 {parquet_file} 时出错: {e}")
                continue
    
    print(f"从ceval-exam加载了 {len(texts)} 条目标分布数据")
    return texts


def load_sft_data(sft_file: str) -> Tuple[List[str], List[Dict]]:
    """
    加载SFT训练数据
    返回用于向量化的文本列表和原始数据列表
    """
    texts = []
    raw_data = []
    
    with open(sft_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载SFT数据"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                raw_data.append(data)
                
                # 提取对话内容用于向量化
                # 主要使用用户问题作为向量化对象
                conversations = data.get('conversations', [])
                text_parts = []
                
                # 添加system prompt（如果有）
                system = data.get('system', '')
                if system:
                    text_parts.append(system[:200])  # 限制system长度
                
                # 添加对话内容
                for conv in conversations:
                    if conv.get('from') == 'human':
                        text_parts.append(conv.get('value', ''))
                    elif conv.get('from') == 'gpt':
                        # 只取回复的前200字符，避免太长
                        response = conv.get('value', '')[:200]
                        text_parts.append(response)
                
                full_text = "\n".join(text_parts)
                texts.append(full_text)
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                continue
    
    print(f"从SFT数据加载了 {len(texts)} 条训练数据")
    return texts, raw_data


def encode_texts(model: SentenceTransformer, texts: List[str], 
                 batch_size: int = 32, desc: str = "编码中") -> np.ndarray:
    """
    使用embedding模型对文本进行向量化
    """
    print(f"开始对 {len(texts)} 条文本进行向量化...")
    
    # 使用sentence-transformers的encode方法
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2归一化，便于计算余弦相似度
    )
    
    return embeddings


def compute_similarity_scores(
    sft_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    top_k: int = 5,
    contamination_threshold: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算每条SFT数据与目标数据的相似度得分
    
    对每条SFT数据：
    1. 计算与所有目标数据的余弦相似度
    2. 选择top_k个最高分
    3. 计算算术平均作为最终得分
    
    新增“去污染”逻辑：
    - 当某个目标题目与语料的相似度 >= contamination_threshold 时，视为过近样本并剔除
    - 在剩余题目中取 top_k 平均作为最终得分

    Returns:
        scores: 每条SFT数据的平均相似度得分
        kept_counts: 每条SFT数据参与平均的题目数量（<= top_k）
    """
    print(f"计算相似度得分 (top_k={top_k})...")
    
    num_sft = sft_embeddings.shape[0]
    scores = np.full(num_sft, -1.0, dtype=np.float32)
    kept_counts = np.zeros(num_sft, dtype=np.int32)
    
    # 分批处理以避免内存溢出
    batch_size = 1000
    for i in tqdm(range(0, num_sft, batch_size), desc="计算相似度"):
        end_idx = min(i + batch_size, num_sft)
        batch_embeddings = sft_embeddings[i:end_idx]
        
        # 计算余弦相似度 (由于已经L2归一化，直接点积即可)
        # shape: (batch_size, num_targets)
        similarities = np.dot(batch_embeddings, target_embeddings.T)
        
        # 对每条数据选择top_k个最高分（含去污染）
        for j in range(batch_embeddings.shape[0]):
            sim_scores = similarities[j]
            if contamination_threshold is not None:
                valid_scores = sim_scores[sim_scores < contamination_threshold]
            else:
                valid_scores = sim_scores

            if valid_scores.size == 0:
                # 全部被判定为污染，保留默认分数 -1
                continue

            k = min(top_k, valid_scores.size)
            top_k_scores = np.partition(valid_scores, -k)[-k:]
            scores[i + j] = float(np.mean(top_k_scores))
            kept_counts[i + j] = int(k)
    
    return scores, kept_counts


def select_top_n_data(
    raw_data: List[Dict],
    scores: np.ndarray,
    top_n: int = 10000,
) -> Tuple[List[Dict], np.ndarray]:
    """
    根据得分筛选前top_n的数据
    """
    num_samples = len(raw_data)
    num_selected = min(top_n, num_samples)
    
    # 按得分降序排列
    sorted_indices = np.argsort(scores)[::-1]
    selected_indices = sorted_indices[:num_selected]
    
    # 获取筛选后的数据和对应的分数
    selected_data = [raw_data[i] for i in selected_indices]
    selected_scores = scores[selected_indices]
    
    print(f"从 {num_samples} 条数据中筛选出 {num_selected} 条（按相似度Top-N）")
    print(f"得分范围: {selected_scores.min():.4f} - {selected_scores.max():.4f}")
    print(f"平均得分: {selected_scores.mean():.4f}")
    
    return selected_data, selected_scores


def save_filtered_data(
    data: List[Dict],
    scores: np.ndarray,
    output_file: str,
    save_scores: bool = True
):
    """
    保存筛选后的数据
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存JSONL格式的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"筛选后的数据已保存到: {output_file}")
    
    # 保存分数信息（用于分析）
    if save_scores:
        score_file = output_path.with_suffix('.scores.json')
        score_info = {
            'num_samples': len(data),
            'score_min': float(scores.min()),
            'score_max': float(scores.max()),
            'score_mean': float(scores.mean()),
            'score_std': float(scores.std()),
            'scores': scores.tolist()
        }
        with open(score_file, 'w', encoding='utf-8') as f:
            json.dump(score_info, f, indent=2, ensure_ascii=False)
        print(f"分数信息已保存到: {score_file}")


def main():
    parser = argparse.ArgumentParser(description="基于Embedding相似度筛选SFT训练数据")
    parser.add_argument("--embedding_model", type=str, 
                        default="./model/Qwen3-Embedding-0.6B",
                        help="Embedding模型路径")
    parser.add_argument("--ceval_exam_dir", type=str,
                        default="../ceval-exam",
                        help="ceval-exam数据目录")
    parser.add_argument("--sft_train_file", type=str,
                        default="./data/sft/train.jsonl",
                        help="SFT训练数据文件")
    parser.add_argument("--output_file", type=str,
                        default="./data/sft/train_filtered.jsonl",
                        help="筛选后的输出文件")
    parser.add_argument("--top_k", type=int, default=5,
                        help="每条语料选择的相关题目数量")
    parser.add_argument("--top_n", type=int, default=10000,
                        help="按相似度排序后保留的训练样本数")
    parser.add_argument("--contamination_threshold", type=float, default=0.95,
                        help="去污染阈值，相似度>=该值的题目将被剔除")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="向量化时的批次大小")
    parser.add_argument("--cache_dir", type=str, default="./cache/embeddings",
                        help="向量缓存目录")
    parser.add_argument("--use_cache", action="store_true",
                        help="是否使用缓存的向量")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 创建缓存目录
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 缓存文件路径
    target_cache_file = cache_dir / "target_embeddings.npy"
    sft_cache_file = cache_dir / "sft_embeddings.npy"
    sft_data_cache_file = cache_dir / "sft_raw_data.json"
    
    print("=" * 60)
    print("SFT数据筛选 - 基于Embedding相似度")
    print("=" * 60)
    print(f"Embedding模型: {args.embedding_model}")
    print(f"目标数据目录: {args.ceval_exam_dir}")
    print(f"SFT数据文件: {args.sft_train_file}")
    print(f"输出文件: {args.output_file}")
    print(f"Top-K题目数: {args.top_k}")
    print(f"保留样本数: {args.top_n}")
    print(f"去污染阈值: {args.contamination_threshold}")
    print("=" * 60)
    
    # 加载模型
    print("\n1. 加载Embedding模型...")
    model = SentenceTransformer(args.embedding_model, device=args.device)
    print(f"模型加载完成，设备: {args.device}")
    
    # 加载或计算目标数据向量
    print("\n2. 处理目标分布数据 (ceval-exam)...")
    if args.use_cache and target_cache_file.exists():
        print("从缓存加载目标数据向量...")
        target_embeddings = np.load(target_cache_file)
        print(f"加载了 {target_embeddings.shape[0]} 条目标数据向量")
    else:
        target_texts = load_ceval_exam_data(args.ceval_exam_dir)
        target_embeddings = encode_texts(model, target_texts, 
                                         batch_size=args.batch_size,
                                         desc="编码目标数据")
        np.save(target_cache_file, target_embeddings)
        print(f"目标数据向量已缓存到: {target_cache_file}")
    
    # 加载或计算SFT数据向量
    print("\n3. 处理SFT训练数据...")
    if args.use_cache and sft_cache_file.exists() and sft_data_cache_file.exists():
        print("从缓存加载SFT数据向量...")
        sft_embeddings = np.load(sft_cache_file)
        with open(sft_data_cache_file, 'r', encoding='utf-8') as f:
            sft_raw_data = json.load(f)
        print(f"加载了 {sft_embeddings.shape[0]} 条SFT数据向量")
    else:
        sft_texts, sft_raw_data = load_sft_data(args.sft_train_file)
        sft_embeddings = encode_texts(model, sft_texts,
                                      batch_size=args.batch_size,
                                      desc="编码SFT数据")
        np.save(sft_cache_file, sft_embeddings)
        with open(sft_data_cache_file, 'w', encoding='utf-8') as f:
            json.dump(sft_raw_data, f, ensure_ascii=False)
        print(f"SFT数据向量已缓存到: {sft_cache_file}")
    
    # 计算相似度得分
    print("\n4. 计算相似度得分...")
    scores, kept_counts = compute_similarity_scores(
        sft_embeddings, 
        target_embeddings,
        top_k=args.top_k,
        contamination_threshold=args.contamination_threshold,
    )

    decontam_all_filtered = int((kept_counts == 0).sum())
    decontam_partial = int(((kept_counts > 0) & (kept_counts < args.top_k)).sum())
    decontam_full = int((kept_counts == args.top_k).sum())
    
    # 打印得分分布统计
    valid_scores = scores[scores >= 0]
    print(f"\n得分分布统计（去污染后）:")
    if valid_scores.size > 0:
        print(f"  最小值: {valid_scores.min():.4f}")
        print(f"  最大值: {valid_scores.max():.4f}")
        print(f"  平均值: {valid_scores.mean():.4f}")
        print(f"  标准差: {valid_scores.std():.4f}")
        print(f"  中位数: {np.median(valid_scores):.4f}")
    else:
        print("  无有效分数（可能去污染阈值过低）")

    print("\n去污染统计:")
    print(f"  完全无可用题目: {decontam_all_filtered}")
    print(f"  可用题目不足Top-K: {decontam_partial}")
    print(f"  可用题目达到Top-K: {decontam_full}")
    
    # 筛选数据
    print("\n5. 筛选高质量数据...")
    filtered_data, filtered_scores = select_top_n_data(
        sft_raw_data, 
        scores,
        top_n=args.top_n,
    )
    
    # 保存结果
    print("\n6. 保存筛选结果...")
    save_filtered_data(filtered_data, filtered_scores, args.output_file)
    
    print("\n" + "=" * 60)
    print("数据筛选完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
