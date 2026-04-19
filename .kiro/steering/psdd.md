# PSDD — Process-augmented Spec Driven Development

## 정의

Kiro SDD를 기반으로 비즈니스 프로세스 레이어와 UI 명세 레이어를 추가한 확장 방법론.
**도구 독립적**: Kiro, Gemini, Claude, Antigravity 등 어떤 AI 에이전트도 이 규칙을 따른다.

---

## 문서 구조 (진실의 원천)

```
docs/
  requirements.md   — EARS 형식 요구사항 (What)
  biz-process.md    — BPMN 드릴다운 프로세스 (How: 비즈니스)
  uis.md            — 인터페이스 명세 (How: 경계)
  design.md         — 기술 설계 (How: 구현)
  tasks.md          — 구현 체크리스트 (Done/Backlog)
```

---

## 에이전트 행동 규칙

### 세션 시작 시
1. `recall("현재 작업 키워드")` — memory에서 컨텍스트 복원
2. `docs/tasks.md` 확인 — 현재 진행 상태 파악
3. 관련 `docs/*.md` 읽기 — 요구사항/설계 파악

### 작업 중
- 모든 구현은 `tasks.md`의 TASK ID를 기준으로 진행
- 설계 결정은 `design.md`에 반영
- 새 요구사항은 `requirements.md`에 REQ ID로 추가

### 작업 완료 시
- `tasks.md`에서 해당 태스크 `[x]` 체크
- `remember(["결정사항", "버그 원인과 해결책"])` — 중요 내용 저장

---

## 문서 작성 순서 (신규 기능)

```
Feature 정의
  → requirements.md (EARS: When ... the system shall ...)
    → biz-process.md (Process→Task→FG→Step→DetailStep→Logic)
      → uis.md (입출력 스키마, 에러 케이스)
        → design.md (아키텍처, 알고리즘, 데이터 구조)
          → tasks.md (구현 단위 분해, REQ ID 연결)
```

---

## 멀티에이전트 역할 분담

| 역할 | 담당 문서 | 입력 |
|------|----------|------|
| Analyst | requirements.md | Feature 정의 |
| Process Agent | biz-process.md | requirements.md |
| UI Agent | uis.md | biz-process.md |
| Architect | design.md | uis.md + requirements.md |
| Executor | tasks.md 기반 구현 | design.md |

---

## 추적성 규칙

- `biz-process.md`의 각 Task는 REQ ID를 참조한다
- `tasks.md`의 각 TASK는 REQ ID를 참조한다
- 모든 Logic 코드는 해당 Process의 DetailStep으로 역추적 가능해야 한다
