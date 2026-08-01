import xml.etree.ElementTree as ET
import torch
import glob
import time
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
from src.turboquant.memory import TurboDiskStore

# 모델 로드 (메인 프로세스에서 한 번만 수행)
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
DIM = 768

def process_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        records = [p.text.strip() for p in root.iter('paragraph') if p.text and len(p.text) > 5]
        return file_path, records
    except:
        return file_path, []

def optimized_ingestion():
    store = TurboDiskStore(DIM, bits=3)
    xml_files = glob.glob('./data_local/chosun/*.xml')
    
    print(f"Starting optimized ingestion of {len(xml_files)} files...")
    start_time = time.time()
    
    # 1. 파일별 텍스트 추출 (병렬 처리)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_file, xml_files)
    
    all_records = [text for _, records in results for text in records]
    print(f"Extraction complete. Total records: {len(all_records)}. Starting batch embedding...")

    # 2. 배치 단위 임베딩 및 저장 (Batch Embedding으로 속도 극대화)
    batch_size = 500
    for i in range(0, len(all_records), batch_size):
        batch = all_records[i : i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        
        for emb in embeddings:
            store.add(torch.from_numpy(emb))
        
        # 로깅 및 예측
        elapsed = time.time() - start_time
        processed = i + len(batch)
        avg_time = elapsed / processed
        eta = (len(all_records) - processed) * avg_time
        print(f"Processed {processed}/{len(all_records)} | ETA: {eta/60:.2f} min")
            
    print(f"Ingestion complete. Total time: {(time.time() - start_time)/60:.2f} min")
    return store, all_records

if __name__ == "__main__":
    store, all_records = optimized_ingestion()
    
    query = "태종은 정도전을 어떻게 했는가?"
    print(f"\nQuerying: {query}")
    query_vec = torch.from_numpy(model.encode(query))
    results = store.search(query_vec, top_k=3)
    
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {all_records[idx][:100]}...")
