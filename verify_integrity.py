import sqlite3
import numpy as np
import os
from datetime import datetime
from memory_store import MemoryStore
from server import encode_batch
import xml.etree.ElementTree as ET
import glob

# 1. 시스템 초기화 및 데이터 주입 (영속성 확보)
def run_ingest_and_verify():
    db_path = "integrity_test.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    store = MemoryStore(db_path)
    
    # 5개 파일만 선택하여 정밀 주입
    files = glob.glob("data/chosun/*.xml")[:5]
    ingested_count = 0
    for file in files:
        tree = ET.parse(file)
        for level3 in tree.findall(".//level3"):
            text = level3.find(".//paragraph").text
            if text:
                vec = encode_batch([text])[0]
                store.add(text, vec)
                ingested_count += 1
    store.commit()
    
    # 프로세스 종료 시뮬레이션을 위해 스토어 객체 삭제
    del store
    print(f"주입 완료 및 세션 종료: {ingested_count}건")

    # 2. 새로운 인스턴스에서 데이터 영속성 확인
    new_store = MemoryStore(db_path)
    count = new_store._db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    print(f"영속성 확인(DB 레코드 수): {count}건")
    
    # 3. 데이터 인출 정확도 테스트
    # 검색어 선정 (첫 번째 저장된 텍스트의 일부를 키워드로)
    sample_text = new_store._db.execute("SELECT text FROM entries LIMIT 1").fetchone()[0]
    query = sample_text[:10] # 첫 10글자로 검색
    print(f"검색 테스트: '{query}'")
    
    query_vec = encode_batch([query])[0]
    results = new_store.search(query, query_vec, top_k=1)
    
    if results and results[0][0] == sample_text:
        print("검증 통과: 정확한 데이터 인출 성공.")
    else:
        print(f"검증 실패: 예상 결과 '{sample_text}', 실제 결과 '{results[0][0] if results else '없음'}'")

if __name__ == "__main__":
    run_ingest_and_verify()
