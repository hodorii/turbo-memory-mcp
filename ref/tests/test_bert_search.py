import os
import shutil
import torch
from sentence_transformers import SentenceTransformer
from src.turboquant.memory import TurboDiskStore
import xml.etree.ElementTree as ET
import glob

# 한국어 특화 임베딩 모델 로드 (768차원)
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def get_embedding(text):
    # BERT 모델을 통한 의미 벡터 추출
    with torch.no_grad():
        embedding = model.encode(text)
    return torch.from_numpy(embedding)

def full_dataset_semantic_query(query):
    # 모델 출력 차원(768)에 맞게 TurboQuantizer 수정 필요
    # 모델 임베딩 후 바로 사용하므로 차원 768로 초기화
    dim = 768
    storage_dir = "/tmp/test_bert_search"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    store = TurboDiskStore(dim, bits=3, storage_dir=storage_dir)
    
    all_records = []
    xml_files = glob.glob('./data_local/chosun/*.xml')
    
    print(f"Loading {len(xml_files)} files...")
    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for p in root.iter('paragraph'):
                if p.text and len(p.text) > 5:
                    all_records.append(p.text.strip())
        except:
            continue
    
    print(f"Ingesting {len(all_records)} records...")
    for text in all_records:
        store.add(get_embedding(text))
        
    # 질의 수행
    query_vec = get_embedding(query)
    results = store.search(query_vec, top_k=3)
    
    print(f"\nQuery: '{query}'")
    for idx, score in results:
        print(f"Result (Score: {score:.4f}): {all_records[idx][:100]}...")

if __name__ == "__main__":
    # TurboQuantizer 초기화 시 차원을 768로 변경해야 합니다.
    # 기존 TurboMemoryStore는 64로 하드코딩되어 있으므로 살짝 수정이 필요함.
    # 아래는 구조적 확인을 위한 호출입니다.
    full_dataset_semantic_query("태종은 정도전을 어떻게 했는가?")
