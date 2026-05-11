import xml.etree.ElementTree as ET
import glob
import numpy as np
from memory_store import MemoryStore
from server import encode_batch
import json

def ingest_all(directory):
    store = MemoryStore("final_annals.db")
    files = glob.glob(f"{directory}/**/*.xml", recursive=True)
    print(f"총 {len(files)}개의 파일 발견. 주입 시작...")
    
    count = 0
    for file in files:
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            for level3 in root.findall(".//level3"):
                content_elem = level3.find(".//paragraph")
                if content_elem is not None and content_elem.text:
                    text = content_elem.text.strip()
                    vec = encode_batch([text])[0]
                    store.add(text, vec, importance=0.8)
                    count += 1
        except: continue
    store.commit()
    print(f"주입 완료: {count}건")

if __name__ == "__main__":
    ingest_all("data")
EOF
