import xml.etree.ElementTree as ET
import glob
import os
import time
from src.engine.memory_engine import MemoryEngine

def ingest_all_batch():
    engine = MemoryEngine(db_path='sillok_full.db')
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
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                eta = (total_files - (i + 1)) * avg_time
                print(f"[{i+1}/{total_files}] 처리 완료: {os.path.basename(file_path)} | ETA: {eta/60:.2f} min")
        except Exception as e:
            print(f"에러 발생 {file_path}: {e}")
            
    # 남은 데이터 처리
    if text_buffer:
        engine.add_batch(text_buffer)
            
    print(f"전체 데이터 인덱싱 완료. 총 소요 시간: {(time.time() - start_time)/60:.2f} min")
    return engine

if __name__ == "__main__":
    ingest_all_batch()
