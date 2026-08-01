import xml.etree.ElementTree as ET
import glob
import os
import time
from src.engine.memory_engine import MemoryEngine
import sqlite3

def get_db_count(db_path):
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        count = conn.execute('SELECT COUNT(*) FROM memory').fetchone()[0]
        conn.close()
        return count
    return 0

def ingest_all_batch_with_progress():
    db_path = 'sillok_full.db'
    engine = MemoryEngine(db_path=db_path)
    xml_files = glob.glob('./data_local/chosun/*.xml')
    total_files = len(xml_files)
    
    print(f"총 {total_files}개의 실록 파일을 배치 처리합니다.")
    start_time = time.time()
    
    batch_size = 500
    text_buffer = []
    
    for i, file_path in enumerate(xml_files):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for p in root.iter('paragraph'):
                if p.text and len(p.text) > 10:
                    text_buffer.append(p.text.strip())
                    
                    if len(text_buffer) >= batch_size:
                        engine.add_batch(text_buffer)
                        text_buffer = []
            
            # 파일 하나 처리 후 로깅
            current_count = get_db_count(db_path)
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = (total_files - (i + 1)) * avg_time
            
            print(f"[{i+1}/{total_files}] {os.path.basename(file_path)} | 현재 DB 저장 건수: {current_count} | ETA: {eta/60:.2f} min")
        except Exception as e:
            print(f"에러 발생 {file_path}: {e}")
            
    if text_buffer:
        engine.add_batch(text_buffer)
            
    print(f"전체 데이터 인덱싱 완료. 총 소요 시간: {(time.time() - start_time)/60:.2f} min")

if __name__ == "__main__":
    ingest_all_batch_with_progress()
