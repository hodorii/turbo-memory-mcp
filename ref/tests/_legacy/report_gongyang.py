import xml.etree.ElementTree as ET
import torch
import glob
from src.turboquant.memory import TurboDiskStore

def get_embedding(text, dim=64):
    vec = torch.zeros(dim)
    # 텍스트에 포함된 키워드 가중치 기반 해시
    if "공양왕" in text:
        vec[0] = 5.0 # 공양왕 키워드 가중치
    hash_val = sum(ord(c) for c in text)
    torch.manual_seed(hash_val)
    return vec + torch.randn(dim)

def query_gongyang():
    dim = 64
    store = TurboDiskStore(dim, bits=3)
    
    all_records = []
    xml_files = glob.glob('./data_local/chosun/*.xml')
    
    # 공양왕 관련 내용만 필터링하여 저장
    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for p in root.iter('paragraph'):
                if p.text and "공양왕" in p.text:
                    all_records.append(p.text.strip())
        except:
            continue
            
    if not all_records:
        print("공양왕 관련 기록을 찾을 수 없습니다.")
        return

    # 메모리 학습
    for text in all_records:
        store.add(get_embedding(text, dim))
        
    # '공양왕' 관련 요약 요청
    query_vec = get_embedding("공양왕", dim)
    results = store.search(query_vec, top_k=5)
    
    print("## 공양왕 관련 실록 기록 요약 및 근거")
    for idx, score in results:
        print(f"- [근거] {all_records[idx]}")

if __name__ == "__main__":
    query_gongyang()
