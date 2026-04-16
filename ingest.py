"""
Batch ingest memories directly into the store — no MCP round-trips.
Usage: python3 ingest.py memories.json
       echo '["text1","text2"]' | python3 ingest.py
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(__file__))

from memory_store import CompressedMemoryStore
from server import STORE_PATH, DIM, BITS, _get_model
import numpy as np

def ingest(texts: list[str]):
    try:
        store = CompressedMemoryStore.load(STORE_PATH)
        if store.dim != DIM or store.bits != BITS:
            raise ValueError
    except Exception:
        store = CompressedMemoryStore(dim=DIM, bits=BITS)

    model = _get_model()
    # Single batch encode — one forward pass for all texts
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=len(texts) > 10)

    for text, emb in zip(texts, embeddings):
        store.add(text, emb.astype("float32"))

    store.save(STORE_PATH)
    print(f"저장 완료: {len(texts)}개 → 총 {store.stats()['entries']}개, 압축률 {store.stats()['compression_ratio']}x")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        texts = json.loads(open(sys.argv[1]).read())
    else:
        texts = json.loads(sys.stdin.read())
    ingest(texts if isinstance(texts, list) else [texts])
