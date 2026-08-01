import xml.etree.ElementTree as ET
import torch
import glob
import time
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from src.turboquant.memory import TurboDiskStore

model = SentenceTransformer('jhgan/ko-sroberta-multitask')
DIM = 768

def get_embedding(text):
    with torch.no_grad():
        embedding = model.encode(text)
    return torch.from_numpy(embedding)

def process_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        records = []
        for p in root.iter('paragraph'):
            if p.text and len(p.text) > 5:
                records.append(p.text.strip())
        return file_path, records
    except Exception as e:
        return file_path, []

def batch_ingestion_parallel():
    store = TurboDiskStore(DIM, bits=3)
    xml_files = glob.glob('./data_local/chosun/*.xml')
    total_files = len(xml_files)
    
    print(f"Starting parallel ingestion of {total_files} files...")
    start_time = time.time()
    all_records = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, xml_files))
        
    for i, (file_path, records) in enumerate(results):
        for text in records:
            store.add(get_embedding(text))
            all_records.append(text)
        
        # 로깅 및 예측
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = total_files - (i + 1)
        eta = remaining * avg_time
        print(f"[{i+1}/{total_files}] Processed: {os.path.basename(file_path)} | ETA: {eta/60:.2f} min")
            
    print(f"Ingestion complete. Total records: {len(all_records)}")
    return store, all_records

if __name__ == "__main__":
    import os
    store, all_records = batch_ingestion_parallel()
    
    query = "태종은 정도전을 어떻게 했는가?"
    print(f"\nQuerying: {query}")
    query_vec = get_embedding(query)
    results = store.search(query_vec, top_k=3)
    
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {all_records[idx][:100]}...")
