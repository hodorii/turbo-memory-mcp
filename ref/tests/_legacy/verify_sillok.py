import json
import torch
from src.turboquant.memory import TurboDiskStore

# 임베딩 함수 (샘플용: 간단하게 텍스트 길이를 이용한 더미 벡터)
def get_embedding(text, dim=64):
    # 실제 환경에서는 모델의 임베딩을 사용합니다.
    # 여기서는 텍스트 특성을 반영한 결정론적 벡터 생성
    vec = torch.zeros(dim)
    hash_val = sum(ord(c) for c in text)
    torch.manual_seed(hash_val)
    return torch.randn(dim)

def test_sillok_memory():
    dim = 64
    store = TurboDiskStore(dim, bits=3)
    
    with open('./data_local/sillok_sample.json', 'r') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} records.")
    
    # 1. 데이터 저장 (양자화)
    for entry in data:
        vec = get_embedding(entry['text'], dim)
        store.add(vec)
        
    # 2. 검색 검증
    target_entry = data[0]
    target_vec = get_embedding(target_entry['text'], dim)
    
    results = store.search(target_vec, top_k=3)
    
    print(f"Top 3 results for '{target_entry['text'][:20]}...':")
    for idx, score in results:
        print(f"Index: {idx}, Score: {score:.4f}, Text: {data[idx]['text'][:20]}...")
        
    assert results[0][0] == 0
    print("Verification passed!")

if __name__ == "__main__":
    test_sillok_memory()
