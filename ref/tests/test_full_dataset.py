import os
import shutil
import xml.etree.ElementTree as ET
import torch
import glob
from src.turboquant.memory import TurboDiskStore

def get_embedding(text, dim=64):
    vec = torch.zeros(dim)
    hash_val = sum(ord(c) for c in text)
    torch.manual_seed(hash_val)
    return torch.randn(dim)

def full_dataset_ingestion():
    dim = 64
    storage_dir = "/tmp/test_full_dataset"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    store = TurboDiskStore(dim, bits=3, storage_dir=storage_dir)
    
    all_records = []
    xml_files = glob.glob('./data_local/chosun/*.xml')
    print(f"Ingesting {len(xml_files)} files...")
    
    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for p in root.iter('paragraph'):
                if p.text and len(p.text) > 5:
                    all_records.append(p.text.strip())
        except:
            continue
            
    print(f"Total records loaded: {len(all_records)}")
    
    # 메모리 학습
    for text in all_records:
        store.add(get_embedding(text, dim))
        
    # 질의 수행
    query = "정도전은 이방원을"
    query_vec = get_embedding(query, dim)
    results = store.search(query_vec, top_k=3)
    
    print(f"\nQuery: '{query}'")
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {all_records[idx][:100]}...")

if __name__ == "__main__":
    full_dataset_ingestion()
