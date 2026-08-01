import xml.etree.ElementTree as ET
import torch
import os
from src.turboquant.memory import TurboDiskStore

# 간단한 임베딩 함수 (실제 환경에서는 pre-trained 모델 사용 권장)
def get_embedding(text, dim=64):
    vec = torch.zeros(dim)
    # 텍스트 내용을 해시하여 벡터 생성
    hash_val = sum(ord(c) for c in text)
    torch.manual_seed(hash_val)
    return torch.randn(dim)

def load_and_search():
    dim = 64
    store = TurboDiskStore(dim, bits=3)
    
    # 데이터 경로
    file_path = './data_local/chosun/2nd_waa_000.xml'
    
    # 1. XML 파싱
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # 실록 본문 추출 (예시 구조)
    records = []
    for element in root.iter('paragraph'):
        # Join all text content including children (like <index>)
        text = "".join(element.itertext()).strip()
        if text and len(text) > 10:
            records.append(text)
            
    print(f"Loaded {len(records)} records from {file_path}")
    
    # 2. 메모리 저장
    for text in records:
        vec = get_embedding(text, dim)
        store.add(vec)
        
    # 3. 검색 수행
    query = "정도전"
    query_vec = get_embedding(query, dim)
    results = store.search(query_vec, top_k=2)
    
    print(f"\nQuery: '{query}'")
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {records[idx][:100]}...")

if __name__ == "__main__":
    load_and_search()
