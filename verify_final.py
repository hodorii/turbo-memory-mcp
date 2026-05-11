import numpy as np
import time
import os
from memory_store import MemoryStore
from server import encode_batch

def benchmark():
    engine = MemoryStore("final_perf.db")
    # 오차율 테스트 (1000쌍)
    errors = []
    for _ in range(1000):
        v1 = np.random.randn(384).astype(np.float32)
        v2 = np.random.randn(384).astype(np.float32)
        n1, s1 = engine.engine.compress(v1)
        n2, s2 = engine.engine.compress(v2)
        est = engine.engine.estimate(n1, s1, n2, s2)
        errors.append(abs(np.dot(v1, v2) - est) / max(abs(np.dot(v1, v2)), 1e-9))
    
    print(f"평균 오차율: {np.mean(errors):.4%}")
    print(f"메모리 사용량(DB Size): {os.path.getsize('final_perf.db') / 1024 / 1024:.2f} MB")

    # 속도 테스트 (10,000건 주입 후 검색)
    for i in range(10000):
        engine.add(f"데이터 {i}", np.random.randn(384).astype(np.float32))
    
    start = time.time()
    engine.search("데이터", np.random.randn(384).astype(np.float32))
    print(f"검색 속도 (1만건): {(time.time() - start) * 1000:.2f} ms")

benchmark()
