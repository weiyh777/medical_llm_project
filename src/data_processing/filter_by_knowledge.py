import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import glob

def load_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_sft_data(file_path):
    data = []
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            data.append(item)
            # Create text representation for embedding/jaccard
            # Combine user query and assistant response
            conversation = ""
            if "conversations" in item:
                for c in item["conversations"]:
                    conversation += c.get("value", "") + "\n"
            elif "prompt" in item and "answer" in item:
                # Handle possible GRPO/other formats just in case, though SFT usually is conversations
                p = item["prompt"]
                if isinstance(p, list):
                    for msg in p:
                        conversation += msg.get("content", "") + "\n"
                else:
                    conversation += str(p) + "\n"
                conversation += str(item.get("answer", ""))
            texts.append(conversation.lower())
    return data, texts

def load_ceval_texts(data_dir):
    texts = []
    pattern = os.path.join(data_dir, "**", "*_val.parquet")
    files = glob.glob(pattern, recursive=True)
    for f in files:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            q_text = str(row['question']).lower()
            # Jaccard check usually primarily concerns the Question stem, 
            # but sometimes the whole QA pair. 
            # The prompt says "with original question", so let's stick to Question text + Options.
            options = []
            for char in ['A', 'B', 'C', 'D']:
                if pd.notna(row.get(char)):
                    options.append(f"{char}. {row[char]}")
            full_text = q_text + " " + " ".join(options)
            texts.append(full_text.lower())
    return texts

def get_jaccard_similarity(str1, str2):
    # Char-level set 
    s1 = set(str1)
    s2 = set(str2)
    if not s1 or not s2: return 0.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union

def batch_jaccard_check(candidate_texts, ceval_texts, threshold=0.8):
    # Returns indices of candidates to DROP
    # Jaccard Distance < 0.2 means Similarity > 0.8
    drop_indices = set()
    
    # Pre-tokenize (set-ify) for speed
    ceval_sets = [set(t) for t in ceval_texts]
    cand_sets = [set(t) for t in candidate_texts]
    
    for i, c_set in enumerate(cand_sets):
        if not c_set: continue
        for ref_set in ceval_sets:
            if not ref_set: continue
            # Optimization: Check length ratio first
            # If lengths differ vastly, Jaccard can't be high
            if len(c_set) < len(ref_set) * threshold or len(ref_set) < len(c_set) * threshold:
                continue
                
            intersection = len(c_set.intersection(ref_set))
            union = len(c_set.union(ref_set))
            sim = intersection / union
            
            if sim > threshold:
                drop_indices.add(i)
                break
    return drop_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords_file", type=str, required=True)
    parser.add_argument("--sft_file", type=str, required=True)
    parser.add_argument("--ceval_dir", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3") # Default or user provided
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--top_k_per_keyword", type=int, default=5)
    parser.add_argument("--jaccard_threshold", type=float, default=0.8, help="Drop if similarity > this (Distance < 1-this)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    # 1. Load Data
    print("Loading keywords...")
    keywords = load_keywords(args.keywords_file)
    print(f"Loaded {len(keywords)} keywords.")
    
    print("Loading SFT data...")
    sft_data, sft_texts = load_sft_data(args.sft_file)
    print(f"Loaded {len(sft_data)} SFT samples.")
    
    # 2. Embeddings
    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model, device=args.device)
    
    # Check for cache
    cache_dir = "./cache/embeddings_knowledge"
    os.makedirs(cache_dir, exist_ok=True)
    
    sft_emb_path = os.path.join(cache_dir, "sft_embeddings.npy")
    if os.path.exists(sft_emb_path):
        print("Loading cached SFT embeddings...")
        sft_embeddings = np.load(sft_emb_path)
    else:
        print("Encoding SFT data...")
        sft_embeddings = model.encode(sft_texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        np.save(sft_emb_path, sft_embeddings)
        
    print("Encoding keywords...")
    # Keywords are short, encode on fly
    kw_embeddings = model.encode(keywords, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    
    # 3. Retrieval
    print("Retrieving top candidates...")
    # dot product for cosine sim (since normalized)
    # kw (K, D) . sft (N, D)^T -> (K, N)
    # This might be huge if N is large. SFT is usually 100k-1M.
    # If SFT is 200k, K is 1k (keywords), matrix is 200MB floats. manageable.
    # Note: user mentioned "200w database". 2M * 1k = 2B floats = 8GB RAM. Might be tight.
    # Let's use semantic_search utility which handles chunking
    hits = util.semantic_search(kw_embeddings, sft_embeddings, top_k=args.top_k_per_keyword, corpus_chunk_size=50000)
    
    candidate_indices = set()
    for hit_list in hits:
        for hit in hit_list:
            candidate_indices.add(hit['corpus_id'])
            
    print(f"Selected {len(candidate_indices)} unique candidates based on knowledge retrieval.")
    
    # 4. Anti-Cheating Filter
    print("Loading CEval data for anti-cheating check...")
    ceval_texts = load_ceval_texts(args.ceval_dir)
    print(f"Loaded {len(ceval_texts)} CEval text references.")
    
    cand_indices_list = list(candidate_indices)
    cand_texts = [sft_texts[i] for i in cand_indices_list]
    
    print("Running Jaccard filter (Removing High Similarity Samples)...")
    # This checks similarity > threshold (default 0.8 => distance < 0.2)
    drop_local_indices = batch_jaccard_check(cand_texts, ceval_texts, threshold=args.jaccard_threshold)
    
    final_indices = []
    dropped_count = 0
    for i, original_idx in enumerate(cand_indices_list):
        if i in drop_local_indices:
            dropped_count += 1
        else:
            final_indices.append(original_idx)
            
    print(f"Dropped {dropped_count} samples due to similarity with test set.")
    print(f"Final dataset size: {len(final_indices)}")
    
    # 5. Save
    final_data = [sft_data[i] for i in final_indices]
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Saved filtered data to {args.output_file}")

if __name__ == "__main__":
    main()
