"""
Batch ingest memories directly into the store — no MCP round-trips.
Usage: python3 ingest.py memories.json
       echo '["text1","text2"]' | python3 ingest.py
"""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from memory_store import MemoryStore
from server import STORE_PATH, DIM, BITS, encode_batch

def ingest(texts: list[str]):
    store = MemoryStore(path=STORE_PATH, dim=DIM, bits=BITS)
    
    # Single batch encode — one forward pass for all texts
    print(f"Encoding {len(texts)} texts...")
    embeddings = encode_batch(texts)

    print(f"Storing {len(texts)} memories...")
    for text, emb in zip(texts, embeddings):
        store.add(text, emb)

    stats = store.stats()
    print(f"저장 완료: {len(texts)}개 → 총 {stats['entries']}개, 압축률 {stats['compression_ratio']}x")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            texts = json.load(f)
    else:
        texts = json.load(sys.stdin)
    
    ingest(texts if isinstance(texts, list) else [texts])
