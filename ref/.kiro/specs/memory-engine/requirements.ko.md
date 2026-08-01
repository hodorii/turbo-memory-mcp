# memory-engine — 요구사항

## 프로젝트 설명
FAISS(FlatIP 내적 인덱스)와 SQLite 영구 저장소, BGE-M3 문장 임베딩(1024차원)을 결합하여 한국 역사 텍스트에 대한 의미 검색을 제공하는 하이브리드 메모리 엔진.

## 언어
ko

## 이해관계자
- 조선왕조실록을 분석하는 역사학자 및 연구자
- 의미 검색이 가능한 영구 장기 메모리가 필요한 AI 에이전트
- 구조적(SQL) 및 의미(벡터) 검색이 모두 필요한 애플리케이션

## 현재 상황
- **MemoryEngine** (`engine/memory_engine.py`): 단순한 하이브리드 아키텍처:
  - **SQLite**: `id`와 `text` 컬럼으로 영구 저장, `executemany`로 배치 삽입 지원
  - **FAISS IndexFlatIP**: 최대 내적 검색을 위한 Inner Product 인덱스, 1024차원(BGE-M3)
  - **SentenceTransformer (BGE-M3)**: 다국어 임베딩 모델, 모듈 레벨 싱글톤으로 로드
- **수집 파이프라인** (tests/): 조선왕조실록 XML 파싱 및 MemoryEngine 수집 스크립트:
  - `ingest_parallel.py`: 멀티프로세싱 XML 파싱 + 배치 ingest (5000/배치)
  - `ingest_all_batch.py`: 전체 XML 파일 배치 ingest (500/배치)
  - `ingest_with_progress.py`: 진행률 + DB 카운트 추적 변형
  - `ingest_real_data.py`: 5개 XML 샘플 ingest
  - `profile_bge.py`: BGE-M3 임베딩 속도/처리량 프로파일링
- **주요 제약사항**: FAISS 인덱스는 순수 인메모리 — 재시작 시 유지되지 않아 재수집 필요
- **차원 불일치**: `benchmarks.py`는 768d(ko-sroberta), `memory_engine.py`는 1024d(BGE-M3) 사용

## 원하는 변경사항
- 기존 MemoryEngine 구현을 있는 그대로 문서화
- FAISS 인덱스 미유지 문제를 알려진 제약사항으로 표시
- 차원 불일치(768d vs 1024d)를 알려진 문제로 표시
