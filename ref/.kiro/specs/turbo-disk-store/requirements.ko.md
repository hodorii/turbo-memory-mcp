# turbo-disk-store — 요구사항

## 프로젝트 설명
TurboQuantizer V2 양자화를 사용하는 디스크 기반 압축 메모리 저장소. 압축된 벡터(indices + signs + scales)를 numpy memmap을 통해 저장하고 LUT 기반 고속 근사 검색을 제공한다.

## 언어
ko

## 이해관계자
- 전체를 RAM에 로드하지 않고 대규모 벡터 저장이 필요한 시스템
- 10만+ 벡터에 대해 sub-second 의미 검색이 필요한 애플리케이션
- 제한된 메모리의 엣지/임베디드 환경

## 현재 상황
- **TurboDiskStore** (`memory.py`): numpy memmap으로 압축 벡터(uint8 indices, int8 signs, float16 scales) 저장, 최대 100만 용량. 검색은 LUT 외적 트릭(O(dim * levels) 점수 계산) + 1-bit 잔차 보정 사용.
- **TurboMemoryStore (없음)**: 16개 테스트 파일이 `memory.py`에서 `TurboMemoryStore`를 import하지만 해당 클래스는 존재하지 않음. 리팩토링 과정에서 `TurboDiskStore`로 대체되며 사라진 것으로 추정. 해당 테스트는 모두 깨짐.
- **테스트 커버리지**: `test_sanity.py`(기본 add/search), `benchmark_turbo_disk.py`(100K 성능), `verify_efficiency.py`(압축비 확인)

## 원하는 변경사항
- 기존 TurboDiskStore 구현 문서화
- 누락된 TurboMemoryStore를 알려진 결함으로 표시
- 새로운 구현 없음 — 현재 상태 기록
