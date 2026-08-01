import xml.etree.ElementTree as ET
import torch
from src.turboquant.memory import TurboDiskStore

def get_embedding(text, dim=64):
    vec = torch.zeros(dim)
    hash_val = sum(ord(c) for c in text)
    torch.manual_seed(hash_val)
    return torch.randn(dim)

def search_across_files():
    dim = 64
    store = TurboDiskStore(dim, bits=3)
    
    # 검색된 타겟 파일
    target_files = ['./data_local/chosun/2nd_waa_107.xml', './data_local/chosun/2nd_wca_116.xml']
    
    all_records = []
    for file_path in target_files:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for p in root.iter('paragraph'):
            if p.text and len(p.text) > 5:
                all_records.append(p.text.strip())
    
    for text in all_records:
        store.add(get_embedding(text, dim))
        
    query = "태종 정도전"
    results = store.search(get_embedding(query, dim), top_k=3)
    
    print(f"Query: '{query}'")
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {all_records[idx][:100]}...")

if __name__ == "__main__":
    search_across_files()
