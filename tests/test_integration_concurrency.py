import threading
import numpy as np
from server import _store, encode

def worker(thread_id):
    for i in range(10):
        text = f"Thread {thread_id} memory {i}"
        embedding = encode(text)
        _store.add(text, embedding, importance=0.5)
        print(f"Thread {thread_id} added {i}")

def test_concurrent_writes():
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Verify total entries
    stats = _store.stats()
    print(f"Total entries: {stats['total_entries']}")
    assert stats['total_entries'] >= 50

if __name__ == "__main__":
    test_concurrent_writes()
