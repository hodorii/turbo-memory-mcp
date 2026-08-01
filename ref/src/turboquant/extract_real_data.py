import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def extract_vectors():
    # 1. 데이터 로드
    with open('data/sillok_sample.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    print(f"Extracted {len(texts)} texts from sample.json")
    
    # 2. BGE-M3 모델로 임베딩 생성
    model = SentenceTransformer('BAAI/bge-m3')
    embeddings = model.encode(texts).astype('float32')
    
    # 3. L2 정규화
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms
    
    # 4. 저장
    np.save('data/real_vectors.npy', embeddings)
    print(f"Saved embeddings to data/real_vectors.npy. Shape: {embeddings.shape}")

if __name__ == "__main__":
    extract_vectors()
