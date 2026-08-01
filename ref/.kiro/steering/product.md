# Product: TurboQuant

## Vision
고성능 벡터 압축 및 의미 검색 엔진. 대규모 임베딩 데이터를 3-bit 양자화로 압축 저장하고, LUT 기반 고속 검색 및 FAISS-SQLite 하이브리드 아키텍처로 실시간 의미 검색을 제공한다.

## Target Users
- 연구자: 대규모 말뭉치 의미 검색 필요
- AI 에이전트: 장기 메모리 저장/검색
- 역사 데이터 분석가: 조선왕조실록 등 대규모 역사 기록 탐색

## Core Metrics
- 압축률: 3-bit 양자화로 Float32 대비 >10x 압축
- 검색 속도: 100K 벡터 기준 sub-second 검색
- Recall@3: 0.67 이상 (1-bit 잔차 보정 적용)
- 인덱싱 처리량: 20만 건 이상 5분 내외
