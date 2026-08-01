# TurboQuant 기억 확장 에이전트 개발 로그

## 2026-05-09: EDEN V3 + Bit-Packing 전환 완료
- **목표:** TurboQuant의 S=1 고정 문제를 EDEN(ICML 2022) 알고리즘으로 대체하고, 저장 효율화를 위한 bit-packing 구현.
- **성과:**
  1. **EDEN V3 (eden.py):** 최적 S 계수(S_bias/S_unbias), Beta(alpha,alpha) Lloyd-Max 코드북, DRIVE 1-bit residual 도입.
  2. **정밀도:** MSE V2 대비 **-96.7%** (d=1024, 3-bit), recall@5 **91.0%** (실록 BGE-M3, V2 65% 대비 +26%p).
  3. **Bit-Packing:** vectorized torch pack/unpack 구현. 인덱스 2-4bit + 부호 1bit 압축. 저장공간 **75%** 절감.
  4. **TurboDiskStore 연동:** packed mmap 저장 + LUT 검색. EDEN/V2 자동 감지(`use_eden`). 검색 속도 동등.
  5. **검증:** 15개 단위 테스트 + 실록 E2E 테스트 통과. 2/3/4비트 전 영역 검증 완료.
- **기술 부채:**
  - bit-packing add()가 per-vector 호출 → 배치 pack 도입 필요
  - sign + index interleave packing (현재는 분리 저장) → 검색 시 locality 개선 가능

## 2026-04-21: 하이브리드 메모리 아키텍처 완성
- **목표:** 조선왕조실록 데이터를 이용한 고성능 기억(Memory) 시스템 프로토타입 구축.
- **성과:** 
  1. **압축 엔진(TurboQuant):** 논문(arXiv:2504.19874) 기반의 3-bit 스칼라 양자화 및 1-bit 잔차 보정 기법 실증 완료. Recall@3 0.67 달성.
  2. **하이브리드 아키텍처:** 
     - **SQLite:** 32만 건 이상의 기록을 영구 저장하는 물리적 장기 기억(Long-term Storage).
     - **FAISS (FlatIP):** 1024차원(BGE-M3) 임베딩 기반의 초고속 의미 검색 인덱스(Semantic Index).
  3. **병렬 배치 처리:** `Multiprocessing` 및 배치(Batch) 임베딩을 통해 20만 건 이상의 실록 데이터를 5분 내외로 인덱싱하는 파이프라인 최적화.

- **향후 과제:**
  - 지식 그래프(Knowledge Graph) 연동을 통한 인과관계(태종-정도전 등) 추론 강화.
  - LLM 에이전트 인터페이스와 기억 엔진 간의 실시간 QA 파이프라인 결합.
  - 데이터 증설에 따른 동적 코드북 재학습(Distribution Adaptation) 루틴 정교화.
