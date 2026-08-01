import xml.etree.ElementTree as ET
import torch
import glob
from sentence_transformers import SentenceTransformer
from src.turboquant.memory import TurboDiskStore

# 모델 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
DIM = 768

def get_embedding(text):
    with torch.no_grad():
        embedding = model.encode(text)
    return torch.from_numpy(embedding)

def ingest_data(store):
    xml_files = glob.glob('./data_local/chosun/*.xml')
    all_records = []
    
    print(f"Starting ingestion of {len(xml_files)} files...")
    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            count_in_file = 0
            for p in root.iter('paragraph'):
                if p.text and len(p.text) > 5:
                    text = p.text.strip()
                    all_records.append(text)
                    store.add(get_embedding(text))
                    count_in_file += 1
            print(f"File processed: {file_path} ({count_in_file} records)")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    print(f"Ingestion complete. Total records: {len(all_records)}")
    return all_records

def verify_query(store, all_records, query):
    print(f"\nVerifying query: '{query}'")
    query_vec = get_embedding(query)
    results = store.search(query_vec, top_k=3)
    
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {all_records[idx][:100]}...")

if __name__ == "__main__":
    store = TurboDiskStore(DIM, bits=3)
    all_records = ingest_data(store)
    verify_query(store, all_records, "태종은 정도전을 어떻게 했는가?")
