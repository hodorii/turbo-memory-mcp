import xml.etree.ElementTree as ET
import glob
from memory_store import MemoryStore
from server import encode_batch

store = MemoryStore("history_final.db")
files = glob.glob("data/chosun/*.xml")
for file in files:
    try:
        tree = ET.parse(file)
        for level3 in tree.findall(".//level3"):
            content = level3.find(".//paragraph").text
            if content:
                vec = encode_batch([content])[0]
                store.add(content, vec, importance=0.8)
    except: continue
store.commit()

# 정확도 검증
query = "태조"
res = store.search(query, encode_batch([query])[0], top_k=3)
print(f"--- 실록 검색 정확도: {query} ---")
[print(f" - {r[0]}") for r in res]
