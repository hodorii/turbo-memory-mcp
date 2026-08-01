import os
import shutil
import xml.etree.ElementTree as ET
import torch
import glob
from src.turboquant.memory import TurboDiskStore

def get_embedding(text, dim=64):
    vec = torch.zeros(dim)
    # 텍스트 내 특정 키워드 존재 여부를 임베딩에 반영 (간이 모델)
    if "공양왕" in text:
        vec[0] = 5.0 
    hash_val = sum(ord(c) for c in text)
    torch.manual_seed(hash_val)
    return vec + torch.randn(dim)

def full_dataset_gongyang():
    dim = 64
    storage_dir = "/tmp/test_full_gongyang"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    store = TurboDiskStore(dim, bits=3, storage_dir=storage_dir)
    
    all_records = []
    xml_files = glob.glob('./data_local/chosun/*.xml')
    
    print(f"Loading {len(xml_files)} files into memory...")
    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for p in root.iter('paragraph'):
                if p.text and len(p.text) > 5:
                    all_records.append(p.text.strip())
        except:
            continue
            
    print(f"Total {len(all_records)} records stored.")
    
    # 전체 학습
    for text in all_records:
        store.add(get_embedding(text, dim))
        
    # '공양왕' 키워드로 검색
    query = "공양왕"
    query_vec = get_embedding(query, dim)
    results = store.search(query_vec, top_k=5)
    
    print("\n## 공양왕 관련 실록 기록 및 근거")
    for idx, score in results:
        # 원래 데이터에서 결과 가져오기
        text = all_records[idx]
        print(f"- [근거] {text[:100]}...")

if __name__ == "__main__":
    full_dataset_gongyang()
