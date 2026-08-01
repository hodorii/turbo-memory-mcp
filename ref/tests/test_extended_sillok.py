import json
import torch
from src.turboquant.memory import TurboDiskStore

# 1. 정도전-이방원 관련 역사적 사실을 포함한 가상 데이터 생성
extended_data = [
    {"text": "정도전은 조선 건국의 핵심 설계자로서 이방원을 견제하였다.", "importance": 0.9},
    {"text": "이방원은 왕자의 난을 일으켜 정도전을 제거하였다.", "importance": 0.9},
    {"text": "정도전은 재상 중심의 정치를 지향하였으나 이방원과 대립하였다.", "importance": 0.8},
    {"text": "조선의 제3대 왕 태종, 왕자의 난을 통해 즉위함.", "importance": 0.5}
]

# 임베딩 함수 (샘플용: 텍스트 의미를 반영하는 간단한 더미 임베딩)
def get_embedding(text, dim=64):
    # 간단한 해시 기반 벡터 생성
    vec = torch.zeros(dim)
    hash_val = sum(ord(c) for c in text)
    torch.manual_seed(hash_val)
    return torch.randn(dim)

def test_extended_memory():
    dim = 64
    store = TurboDiskStore(dim, bits=3)
    
    # 데이터 학습
    for entry in extended_data:
        vec = get_embedding(entry['text'], dim)
        store.add(vec)
        
    # 검색 수행
    query = "정도전은 이방원을"
    query_vec = get_embedding(query, dim)
    results = store.search(query_vec, top_k=2)
    
    print(f"Query: '{query}'")
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {extended_data[idx]['text']}")

if __name__ == "__main__":
    test_extended_memory()
