import xml.etree.ElementTree as ET
import torch
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from src.turboquant.memory import TurboDiskStore

# 1. 모델 설정
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
DIM = 768

def get_embedding(text):
    with torch.no_grad():
        embedding = model.encode(text)
    return torch.from_numpy(embedding)

def batch_ingestion():
    store = TurboDiskStore(DIM, bits=3)
    
    xml_files = glob.glob('./data_local/chosun/*.xml')
    all_records = []
    
    print(f"Starting batch ingestion for {len(xml_files)} files...")
    
    count = 0
    batch_size = 1000
    
    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for p in root.iter('paragraph'):
                if p.text and len(p.text) > 5:
                    text = p.text.strip()
                    all_records.append(text)
                    
                    # 2. 배치 단위 학습 및 로깅
                    store.add(get_embedding(text))
                    count += 1
                    
                    if count % batch_size == 0:
                        print(f"Processed {count} records...")
        except:
            continue
            
    print(f"Ingestion complete. Total: {count} records.")
    
    # 3. 질문 검증
    query = "태종은 정도전을 어떻게 했는가?"
    results = store.search(get_embedding(query), top_k=3)
    
    print(f"\nQuery: '{query}'")
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {all_records[idx][:100]}...")

if __name__ == "__main__":
    batch_ingestion()
