import xml.etree.ElementTree as ET
import torch
from src.turboquant.memory import TurboDiskStore

def get_embedding(text, dim=64):
    vec = torch.zeros(dim)
    hash_val = sum(ord(c) for c in text)
    torch.manual_seed(hash_val)
    return torch.randn(dim)

def search_relation():
    dim = 64
    store = TurboDiskStore(dim, bits=3)
    file_path = './data_local/chosun/2nd_waa_000.xml'
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    records = [p.text.strip() for p in root.iter('paragraph') if p.text and len(p.text) > 5]
    
    for text in records:
        store.add(get_embedding(text, dim))
        
    query = "태종 이성계"
    results = store.search(get_embedding(query, dim), top_k=3)
    
    print(f"Query: '{query}'")
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {records[idx][:100]}...")

if __name__ == "__main__":
    search_relation()
