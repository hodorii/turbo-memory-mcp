import os
import numpy as np
from memory_store import MemoryStore
from server import encode_batch, encode

def validate():
    db_path = "test_hybrid.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    store = MemoryStore(path=db_path)
    
    # 1. 데이터 준비 (의미는 비슷하지만 키워드가 명확히 다른 기술적 문서들)
    documents = [
        "이것은 첫 번째 작업인 TASK-01에 대한 명세서입니다.",
        "이것은 두 번째 작업인 TASK-02에 대한 명세서입니다.",
        "TurboQuant 알고리즘은 벡터를 3-bit로 압축합니다.",
        "QJL 보정은 내적 추정의 편향을 제거합니다.",
        "에이전트가 메모리를 기억(remember)하고 회상(recall)합니다."
    ]
    
    print("--- 1. 데이터 저장 (Hybrid Indexing) ---")
    embs = encode_batch(documents)
    for t, e in zip(documents, embs):
        store.add(t, e)
    
    # 2. 정확한 심볼 매칭 테스트 (BM25의 힘)
    print("\n--- 2. 정확한 키워드 매칭 테스트 (Query: 'TASK-01') ---")
    # 'TASK-01'과 'TASK-02'는 벡터 공간에서 매우 가깝지만, FTS5는 다르게 인식해야 함
    q_text = "TASK-01"
    q_vec = encode(q_text)
    results = store.search(q_text, q_vec, top_k=2)
    
    for r in results:
        print(f"ID: {r[0]}, Score: {r[2]:.4f}, Text: {r[1]}")
    
    assert "TASK-01" in results[0][1], "FTS5 should rank TASK-01 first via keyword match"

    # 3. 불용어 필터링 테스트
    print("\n--- 3. 불용어 필터링 테스트 (Query: '알고리즘은') ---")
    # '은'이 제거되고 '알고리즘'만 남아서 매칭되어야 함
    q_text = "알고리즘은"
    q_vec = encode(q_text)
    results = store.search(q_text, q_vec, top_k=1)
    print(f"Result: {results[0][1]} (Score: {results[0][2]:.4f})")
    assert "TurboQuant" in results[0][1]

    # 4. 복합 검색 테스트 (Vector + Keyword)
    print("\n--- 4. 하이브리드 결합 테스트 (Query: '벡터 보정') ---")
    # '벡터'는 TurboQuant 문서에 있고, '보정'은 QJL 문서에 있음
    # 하이브리드 점수가 두 가지 맥락을 모두 고려하는지 확인
    q_text = "벡터 보정"
    q_vec = encode(q_text)
    results = store.search(q_text, q_vec, top_k=2)
    for r in results:
        print(f"ID: {r[0]}, Score: {r[2]:.4f}, Text: {r[1]}")

    if os.path.exists(db_path): os.remove(db_path)

if __name__ == "__main__":
    validate()
