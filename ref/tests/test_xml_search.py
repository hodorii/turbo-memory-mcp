import os
import shutil
import xml.etree.ElementTree as ET
import torch
from src.turboquant.memory import TurboDiskStore

# 텍스트 벡터화 함수 (샘플링)
def get_embedding(text, dim=64):
    vec = torch.zeros(dim)
    # 텍스트의 간단한 문자열 통계 기반 해시
    hash_val = sum(ord(c) for c in text)
    torch.manual_seed(hash_val)
    return torch.randn(dim)

def search_xml_data(query_text):
    dim = 64
    storage_dir = "/tmp/test_xml_search"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    store = TurboDiskStore(dim, bits=3, storage_dir=storage_dir)
    
    # 데이터 경로
    file_path = './data_local/chosun/2nd_waa_000.xml'
    
    # XML 파싱
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # <paragraph> 태그 안의 한문 텍스트 추출
    records = []
    for element in root.iter('paragraph'):
        text = element.text
        if text and len(text) > 5:
            records.append(text.strip())
            
    print(f"Loaded {len(records)} records.")
    
    # 메모리 학습
    for text in records:
        vec = get_embedding(text, dim)
        store.add(vec)
        
    # 질의 수행
    query_vec = get_embedding(query_text, dim)
    results = store.search(query_vec, top_k=3)
    
    print(f"\nQuery: '{query_text}'")
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {records[idx][:50]}...")

if __name__ == "__main__":
    search_xml_data("太祖") # '태조'에 해당하는 한문
