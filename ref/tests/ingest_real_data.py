import xml.etree.ElementTree as ET
import glob
from src.engine.memory_engine import MemoryEngine

def ingest_chosun_sillok():
    engine = MemoryEngine(db_path='sillok.db')
    xml_files = glob.glob('./data_local/chosun/*.xml')
    
    print(f"총 {len(xml_files)}개의 실록 파일을 처리합니다.")
    
    # 1. 파일별 파싱 및 저장 (샘플링)
    for file_path in xml_files[:5]: 
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            count = 0
            for p in root.iter('paragraph'):
                if p.text and len(p.text) > 10:
                    engine.add(p.text.strip())
                    count += 1
            print(f"완료: {file_path} ({count}개 레코드 삽입)")
        except Exception as e:
            print(f"에러: {file_path} - {e}")
            
    return engine

if __name__ == "__main__":
    engine = ingest_chosun_sillok()
    query = "정도전과 이방원의 관계"
    print(f"\n질의: {query}")
    for text, score in engine.search(query):
        print(f"[{score:.4f}] {text}")
