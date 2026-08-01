# sillok-ingestion — 요구사항

## 프로젝트 설명
조선왕조실록 XML 데이터를 파싱, 임베딩, 인덱싱하여 TurboQuant 저장 시스템에 수집하는 데이터 파이프라인. TurboDiskStore와 MemoryEngine 백엔드를 모두 지원한다.

## 언어
ko

## 이해관계자
- 대규모 역사 텍스트 말뭉치를 질의해야 하는 연구자
- 674+ XML 파일에서 32만+ 레코드를 수집하는 시스템
- 종단간 검색 정확도를 검증하는 QA 엔지니어

## 현재 상황
- **데이터 소스**: `data_local/chosun/` 디렉토리의 674개 XML 파일(조선왕조실록 단락), `danjong/`, `sillok/` 디렉토리 추가
- **수집 방식**:
  1. **MemoryEngine 경로**: XML 파싱 → BGE-M3 임베딩 → FAISS 인덱스 + SQLite 저장 (`ingest_parallel.py`, `ingest_all_batch.py`)
  2. **TurboDiskStore 경로**: XML 파싱 → ko-sroberta 임베딩 → TurboQuantizer_V2 양자화 → memmap 저장 (`optimized_ingestion.py`, `batch_ingestion_v2.py`)
  3. **TurboMemoryStore 경로**: (깨짐) 존재하지 않는 `TurboMemoryStore` 클래스를 참조하는 레거시 스크립트
- **병렬 처리**: XML 파싱에 멀티프로세싱(Pool), 모델 추론에 배치 임베딩
- **성능**: BGE-M3 배치 임베딩으로 20만+ 레코드를 약 5분에 인덱싱
- **테스트 커버리지**: 16개 이상 수집/질의 스크립트, 대부분 하드코딩된 질의 검증 포함 ("태종은 정도전을 어떻게 했는가?" 등)

## 원하는 변경사항
- 기존 모든 수집 파이프라인 변형 문서화
- 깨진 TurboMemoryStore 스크립트를 알려진 결함으로 표시
- 새로운 구현 없음 — 현재 상태 기록
