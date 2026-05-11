# TurboMemory-MCP

**지능형 기억 관리 엔진: 조선왕조실록 및 역사적 맥락 학습 에이전트**

이 프로젝트는 대규모 역사 데이터(조선왕조실록 등)를 기억하고, 지능적으로 관리(망각 및 침전)하며, 정확하게 인출하는 기억 엔진을 구축합니다.

## 아키텍처 원칙 (First Principles)
1. **데이터 정합성 우선**: 압축률보다 데이터의 물리적 무결성(FP32)과 차원 정합성을 최우선시합니다.
2. **이원화 구조 (Hybrid Storage)**: 
   - **SQLite**: 팩트(키워드), 시간, 중요도 등의 메타데이터 저장 (SSoT).
   - **TurboQuant**: 벡터 의미론적 맥락 저장 및 내적 추정(Inner Product Estimation).
3. **지능형 기억 관리 (Sedimentation)**: 모든 기억은 수명과 중요도에 따라 'Working'에서 'Archive'로 침전(Sediment)되어 시스템 효율을 최적화합니다.

## 핵심 기술 스택
- **Search**: Hybrid Search (FTS5 키워드 검색 + TurboQuant 벡터 유사도).
- **Intelligence**: 에빙하우스 망각 곡선 기반의 Scoring 엔진.
- **Data Integrity**: 명시적 차원 강제(Reshape)를 통한 차원 오염 원천 차단.
