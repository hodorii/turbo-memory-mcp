import xml.etree.ElementTree as ET
import numpy as np
from memory_store import MemoryStore
from server import encode_batch
import os

def validate_pipeline():
    db_path = "final_annals.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    store = MemoryStore(db_path)
    
    # 1. 샘플 XML 파일 파싱 (2nd_waa_000.xml 사용)
    file = "data/chosun/2nd_waa_000.xml"
    tree = ET.parse(file)
    # level3의 첫 번째 항목 추출
    level3 = tree.findall(".//level3")[0]
    paragraph = level3.find(".//paragraph")
    if paragraph is None:
        print("파싱 실패: paragraph 태그를 찾을 수 없습니다.")
        return
    raw_text = "".join(paragraph.itertext()).strip()
    
    print(f"--- 1. 샘플링 추출 내용 ---\n{raw_text[:200]}...\n")
    
    # 2. 주입 (바인딩)
    vec = encode_batch([raw_text])[0]
    store.add(raw_text, vec)
    store.commit()
    print("--- 2. 주입 및 바인딩 완료 ---")
    
    # 3. 임베딩 후 내용 검색
    query = raw_text[:30] # 텍스트 일부로 검색
    print(f"검색 쿼리: '{query}'")
    
    query_vec = encode_batch([query])[0]
    results = store.search(query, query_vec, top_k=1)
    
    # 4. 최종 정확도 확인
    if results and results[0][0] == raw_text:
        print("\n--- 3. 최종 결과: 완벽 매칭 성공 ---")
        print(f"매칭된 텍스트: {results[0][0][:100]}...")
    else:
        print(f"\n--- 3. 최종 결과: 매칭 실패 ---")
        if results:
            print(f"실제 결과: {results[0][0][:100]}...")

if __name__ == "__main__":
    validate_pipeline()
