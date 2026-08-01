import xml.etree.ElementTree as ET
import glob
import os
import time
import multiprocessing as mp
from src.engine.memory_engine import MemoryEngine
import sqlite3

# 멀티프로세스 환경에서 사용할 수 있도록 MemoryEngine을 
# 프로세스 간 공유 가능한 구조로 변경하거나, 
# 파일 단위로 병렬 파싱 후 마지막에 일괄 삽입하는 방식이 가장 안전함.

def parse_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        records = [p.text.strip() for p in root.iter('paragraph') if p.text and len(p.text) > 10]
        return records
    except:
        return []

def ingest_parallel():
    db_path = 'sillok_full.db'
    engine = MemoryEngine(db_path=db_path)
    xml_files = glob.glob('./data_local/chosun/*.xml')
    total_files = len(xml_files)
    
    print(f"Starting parallel ingestion of {total_files} files...")
    start_time = time.time()
    
    # 1. 병렬 파일 파싱
    with mp.Pool(processes=mp.cpu_count()) as pool:
        all_results = pool.map(parse_file, xml_files)
    
    # 2. 결과물 평탄화 및 배치 저장
    all_texts = [text for sublist in all_results for text in sublist]
    
    print(f"Parsing complete. Ingesting {len(all_texts)} records into SQLite/FAISS...")
    
    batch_size = 5000
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        engine.add_batch(batch)
        print(f"Processed {min(i + batch_size, len(all_texts))}/{len(all_texts)} records...")
            
    print(f"Total time: {(time.time() - start_time)/60:.2f} min")

if __name__ == "__main__":
    ingest_parallel()
