import torch
import time
from sentence_transformers import SentenceTransformer
import psutil
import os

def get_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def profile_bge_m3():
    print("--- BGE-M3 Profiling ---")
    
    # 1. Profile Model Loading
    print("Loading BGE-M3 model...")
    start_load = time.time()
    model = SentenceTransformer('BAAI/bge-m3')
    load_time = time.time() - start_load
    print(f"Model Load Time: {load_time:.2f} seconds")
    print(f"RAM Usage after Load: {get_mem_usage():.2f} MB")
    
    # 2. Profile Embedding Generation (Inference)
    # Test with different sample sizes
    sample_sizes = [1, 10, 50]
    test_text = "이것은 모델 성능 측정을 위한 샘플 텍스트입니다. 조선왕조실록 데이터를 처리하는 속도를 가늠해봅니다."
    
    print("\n--- Inference Profiling ---")
    for size in sample_sizes:
        texts = [test_text] * size
        
        # Warm up
        if size == 1:
            model.encode([test_text], batch_size=1)
            
        start_inf = time.time()
        # Using batch_size=1 to avoid OOM on some environments during profiling
        model.encode(texts, batch_size=1, show_progress_bar=False)
        inf_time = time.time() - start_inf
        
        print(f"Size: {size:3d} | Total: {inf_time:6.2f}s | Per Text: {inf_time/size:6.4f}s")

if __name__ == "__main__":
    profile_bge_m3()
