import xml.etree.ElementTree as ET
import glob
import numpy as np
from memory_store import MemoryStore
from server import encode_batch
import os

def ingest_xml_data(directory):
    store = MemoryStore("history_full.db")
    files = glob.glob(f"{directory}/*.xml")
    
    count = 0
    for file in files:
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            
            for level3 in root.findall(".//level3"):
                title_elem = level3.find(".//mainTitle")
                content_elem = level3.find(".//paragraph")
                
                if title_elem is None or content_elem is None or title_elem.text is None or content_elem.text is None:
                    continue
                    
                title = title_elem.text.strip()
                content = content_elem.text.strip()
                
                # 메타데이터 추출
                subjects = [s.text for s in level3.findall(".//subjectClass")]
                
                # 중요도 설정
                importance = 0.9 if any(s in ['왕실', '변란', '역사'] for s in subjects) else 0.5
                
                # 주입
                text = f"{title}: {content}"
                vec = encode_batch([text])[0]
                store.add(text, vec, importance=importance)
                count += 1
                if count % 100 == 0:
                    print(f"주입 중... {count}건")
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            
    store.commit()
    print(f"조선왕조실록 데이터 {count}건 주입 완료.")

if __name__ == "__main__":
    ingest_xml_data("data/chosun")
