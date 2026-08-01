import json
import torch
from src.turboquant.memory import TurboDiskStore

# 임베딩 함수: 텍스트 특성을 벡터로 변환
def get_embedding(text, dim=64):
    vec = torch.zeros(dim)
    hash_val = sum(ord(c) for c in text)
    torch.manual_seed(hash_val)
    return torch.randn(dim)

def search_sillok(query_text):
    dim = 64
    store = TurboDiskStore(dim, bits=3)
    
    with open('./data_local/sillok_sample.json', 'r') as f:
        data = json.load(f)
        
    # 1. 데이터 저장
    for entry in data:
        vec = get_embedding(entry['text'], dim)
        store.add(vec)
        
    # 2. 질문에 대한 임베딩 생성 및 검색
    query_vec = get_embedding(query_text, dim)
    results = store.search(query_vec, top_k=3)
    
    print(f"Query: '{query_text}'")
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {data[idx]['text']}")

if __name__ == "__main__":
    search_sillok("성삼문")
