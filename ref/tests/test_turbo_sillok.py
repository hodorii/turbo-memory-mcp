import xml.etree.ElementTree as ET
import torch
import os
import glob
import time
import sys
from sentence_transformers import SentenceTransformer
from src.turboquant.memory import TurboDiskStore

def extract_texts_from_xml(file_paths):
    all_texts = []
    for path in file_paths:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            for element in root.iter('paragraph'):
                text = "".join(element.itertext()).strip()
                if text and len(text) > 10:
                    # Truncate to avoid BGE-M3 memory overflow with extremely long paragraphs
                    all_texts.append(text[:512])
        except Exception as e:
            print(f"Error parsing {path}: {e}")
    return all_texts

def run_sillok_test():
    # Setup logging to file
    log_file = open("sillok_test.log", "w", encoding="utf-8")
    
    def log_print(msg):
        print(msg)
        log_file.write(str(msg) + "\n")
        log_file.flush()

    # 1. Setup
    log_print("--- Joseon Wangjo Sillok Real Data Test ---")
    model = SentenceTransformer('BAAI/bge-m3')
    DIM = 1024 # BGE-M3 dimension
    storage_dir = "sillok_turbo_storage"
    
    # Load a sample of XML files from chosun directory
    xml_files = glob.glob('data_local/chosun/*.xml')[:50] # Test with first 50 files
    log_print(f"Loading data from {len(xml_files)} XML files...")
    
    texts = extract_texts_from_xml(xml_files)[:500]
    log_print(f"Extracted {len(texts)} valid paragraphs.")
    
    # 2. Generate Embeddings
    log_print("Generating embeddings (BGE-M3)...")
    start_emb = time.time()
    embeddings = model.encode(texts, batch_size=4, show_progress_bar=True)
    embeddings = torch.from_numpy(embeddings).float()
    log_print(f"Embedding time: {time.time() - start_emb:.2f}s")
    
    # 3. Store in TurboDiskStore
    log_print("Storing in TurboDiskStore...")
    store = TurboDiskStore(DIM, bits=3, storage_dir=storage_dir)
    
    start_store = time.time()
    for i in range(len(embeddings)):
        store.add(embeddings[i])
    log_print(f"Storage time: {time.time() - start_store:.2f}s")
    
    # 4. Search Test
    queries = [
        "정도전의 정치 철학",
        "이방원의 권력 장악",
        "태조 이성계의 가계"
    ]
    
    log_print("\n--- Search Results ---")
    for q in queries:
        log_print(f"\nQuery: {q}")
        q_emb = torch.from_numpy(model.encode([q]).astype('float32'))[0]
        
        start_search = time.time()
        results = store.search(q_emb, top_k=3)
        search_time = time.time() - start_search
        
        log_print(f"Search time: {search_time:.4f}s")
        for idx, score in results:
            if idx < len(texts):
                log_print(f"[{score:.4f}] {texts[idx][:150]}...")
            else:
                log_print(f"Index {idx} out of bounds.")
    
    log_file.close()

if __name__ == "__main__":
    run_sillok_test()
