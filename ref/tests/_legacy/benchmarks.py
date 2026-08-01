import time
import torch
import numpy as np
import os
import psutil
from sentence_transformers import SentenceTransformer
from src.turboquant.memory import TurboDiskStore

# 모델 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
DIM = 768
NUM_LONG_TERM = 50000 # 측정 가능한 규모로 설정 (전체 데이터는 메모리 과부하 우려)
NUM_SHORT_TERM = 1000

def get_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def run_benchmarks():
    print(f"--- 실측 시작 (초기 메모리: {get_mem_usage():.2f} MB) ---")
    
    # 1. 고해상도(Float32) 장기 메모리 준비
    long_term_vecs = [torch.randn(DIM) for _ in range(NUM_LONG_TERM)]
    short_term_vecs = [torch.randn(DIM) for _ in range(NUM_SHORT_TERM)]
    
    # 2. TurboQuant 압축 기반 장기 메모리 실측
    store = TurboDiskStore(DIM, bits=3)
    start_store = time.time()
    for v in long_term_vecs:
        store.add(v)
    store_time = time.time() - start_store
    mem_after_store = get_mem_usage()
    
    # 3. 검색 성능 실측
    query_vec = torch.randn(DIM)
    
    # A. 고해상도 비교 (Short-term)
    start_st = time.time()
    # 텐서로 모아서 연산 (최적화)
    short_term_tensor = torch.stack(short_term_vecs)
    st_sims = torch.nn.functional.cosine_similarity(query_vec.unsqueeze(0), short_term_tensor)
    st_time = time.time() - start_st
    
    # B. TurboQuant 압축 검색 (Long-term)
    start_lt = time.time()
    results = store.search(query_vec, top_k=3)
    lt_time = time.time() - start_lt
    
    print(f"--- 실측 결과 ---")
    print(f"장기 메모리({NUM_LONG_TERM}개) 저장 시간: {store_time:.2f}초")
    print(f"저장 후 메모리 사용량: {mem_after_store:.2f} MB")
    print(f"단기 메모리(Float32, {NUM_SHORT_TERM}개) 검색 속도: {st_time:.4f}초")
    print(f"장기 메모리(3-bit VQ, {NUM_LONG_TERM}개) 검색 속도: {lt_time:.4f}초")
    print(f"검색 효율 비교: 3-bit VQ가 단기 메모리보다 { (st_time / (lt_time/(NUM_LONG_TERM/NUM_SHORT_TERM))) if lt_time > 0 else 0 :.2f}배 빠름")

if __name__ == "__main__":
    run_benchmarks()
