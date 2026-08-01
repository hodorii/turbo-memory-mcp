# Tech Stack & Architecture

## Language & Runtime
- Python 3.9+ (venv 가상환경)
- No package manager (직접 실행, namespace packages)

## Core Dependencies
- **torch**: 텐서 연산, 양자화/역양자화
- **numpy**: memmap 기반 디스크 저장
- **faiss-cpu**: FAISS FlatIP 인덱스
- **sentence-transformers**: BGE-M3 (1024d) / ko-sroberta (768d) 임베딩

## Architecture

### Two Quantization Algorithms
1. **TurboQuantizer_V1** (`quantizer.py`): 단순 3-bit 스칼라 양자화 + Random Rotation (QR 분해)
2. **TurboQuantizer_V2** (`memory.py`): 3-bit + 1-bit 잔차 보정 (sign + scale)

### Two Storage Systems
1. **TurboDiskStore** (`memory.py`): numpy memmap 기반 디스크 저장, LUT 기반 고속 검색
2. **MemoryEngine** (`engine/memory_engine.py`): FAISS + SQLite + SentenceTransformer 하이브리드

## Key Patterns
- `__init__.py` 없음 → Python 3.3+ namespace packages 의존
- 테스트: 프레임워크 없음, 개별 `python file.py` 실행
- 모듈 임포트: `from src.turboquant...` (루트 디렉토리에서 실행)
- 데이터: `data/`는 `../turbo-memory-mcp/data` 심볼릭 링크

## Known Issues
- **차원 불일치**: benchmarks.py(768d) vs memory_engine.py(1024d)
- **TurboMemoryStore 미존재**: memory.py에 없는 클래스를 16개 테스트가 import → 깨진 상태
- SQLite DB 파일(`*.db`)은 생성 파일로 커밋 금지
