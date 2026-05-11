import sqlite3, xml.etree.ElementTree as ET, glob, numpy as np
from memory_store import MemoryStore
from server import encode_batch

def ingest_and_verify():
    db_path = "final_verification.db"
    store = MemoryStore(db_path)
    files = glob.glob("data/chosun/*.xml")
    
    total_injected = 0
    # 주입
    for file in files:
        tree = ET.parse(file)
        for level3 in tree.findall(".//level3"):
            content = "".join(level3.find(".//paragraph").itertext()).strip()
            if content:
                vec = encode_batch([content])[0]
                store.add(content, vec)
                total_injected += 1
    
    print(f"주입 완료: {total_injected}건")
    
    # 팩트 인출 검증 (이방원 정몽주 관련 사실)
    query = '이방원은 정몽주를'
    vec = encode_batch([query])[0]
    res = store.search(query, vec, top_k=1)
    
    if res and '이방원' in res[0][0] and '정몽주' in res[0][0]:
        print(f"검증 성공: '{res[0][0][:50]}...'")
    else:
        print("검증 실패: 팩트 인출 실패")

ingest_and_verify()
