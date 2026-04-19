# PSDD — Process-augmented Spec Driven Development

> 도구 독립적 명세 주도 개발 방법론  
> Kiro, Gemini, Claude, Antigravity 모두 호환

---

## 정의

Kiro SDD를 기반으로 **비즈니스 프로세스 레이어**와 **UI 명세 레이어**를 추가한 확장 방법론.
코드가 먼저 있고, 문서가 코드를 역방향으로 정제(distill)하는 구조.

---

## 문서 구조 (진실의 원천)

```
docs/
  PSDD.md           ← 방법론 정의 (현재 문서)
  requirements.md   ← EARS 형식 요구사항 (What)
  biz-process.md    ← BPMN 드릴다운 프로세스 (How: 비즈니스)
  uis.md            ← 인터페이스 명세 (How: 경계)
  design.md         ← 기술 설계 (How: 구현)
  tasks.md          ← 구현 체크리스트 (Done/Backlog)
```

각 레이어는 상위 레이어의 "왜"를 "어떻게"로 분해한다.

---

## 에이전트 행동 규칙

### 세션 시작 시

1. **컨텍스트 복원**
   - memory MCP 사용 가능 시: `recall("현재 작업 키워드")`
   - 없으면: `docs/tasks.md` 직접 읽기

2. **현재 상태 파악**
   - `docs/tasks.md` 확인 — 어디까지 완료됐는가?
   - 관련 `docs/*.md` 읽기 — 요구사항/설계 파악

### 작업 중

- 모든 구현은 `tasks.md`의 **TASK ID**를 기준으로 진행
- 설계 결정은 `design.md`에 반영
- 새 요구사항은 `requirements.md`에 **REQ ID**로 추가
- 프로세스 변경은 `biz-process.md`에 반영
- UI 변경은 `uis.md`에 반영

### 작업 완료 시

1. `tasks.md`에서 해당 태스크 `[x]` 체크
2. memory MCP 사용 가능 시: `remember(["결정사항", "버그 원인과 해결책"])`
3. 중요 결정사항은 해당 문서에 반영

---

## 문서 작성 순서 (신규 기능)

```
Feature 정의
  ↓
requirements.md (EARS: When ... the system shall ...)
  ↓
biz-process.md (Process→Task→FG→Step→DetailStep→Logic)
  ↓
uis.md (입출력 스키마, 에러 케이스)
  ↓
design.md (아키텍처, 알고리즘, 데이터 구조)
  ↓
tasks.md (구현 단위 분해, REQ ID 연결)
```

---

## 멀티에이전트 역할 분담

PSDD는 subagent/crew 메커니즘과 자연스럽게 결합된다.

| 역할 | 담당 문서 | 입력 | 출력 |
|------|----------|------|------|
| **Analyst** | requirements.md | Feature 정의 | REQ-XX 목록 |
| **Process Agent** | biz-process.md | requirements.md | P-XX 프로세스 |
| **UI Agent** | uis.md | biz-process.md | UI-XX 인터페이스 |
| **Architect** | design.md | uis.md + requirements.md | 기술 설계 |
| **Executor** | 코드 구현 | design.md + tasks.md | 실제 코드 |

각 에이전트는 상위 문서를 컨텍스트로 주입받고 자신의 레이어만 담당한다.

---

## 추적성 규칙 (Traceability)

모든 산출물은 양방향 추적 가능해야 한다:

- `biz-process.md`의 각 **Task**는 **REQ ID**를 참조
- `tasks.md`의 각 **TASK**는 **REQ ID**를 참조
- 모든 **Logic 코드**는 해당 Process의 **DetailStep**으로 역추적 가능

예시:
```
REQ-01-01 (requirements.md)
  ↓
P-01 / T-01-1 (biz-process.md)
  ↓
TASK-01 (tasks.md)
  ↓
turbo_quant.py::build_state() (실제 코드)
```

---

## 도구별 통합 방법

### Kiro CLI

```json
// .kiro/agents/default.json
{
  "resources": [
    "file://docs/PSDD.md",
    "file://docs/tasks.md"
  ]
}
```

또는 `.kiro/steering/psdd.md`에 이 문서 복사 (자동 로드).

### Gemini CLI

`GEMINI.md` 파일에:
```markdown
# Project Rules

이 프로젝트는 PSDD 방법론을 따릅니다.
세션 시작 시 `docs/PSDD.md`를 읽고 규칙을 따르세요.
```

### Claude Desktop / API

`.claude/project.md` 또는 시스템 프롬프트에:
```markdown
This project follows PSDD methodology.
Read docs/PSDD.md at session start.
```

### Antigravity

세션 시작 시 컨텍스트에 `docs/PSDD.md` 포함.

---

## 핵심 원칙 3가지

1. **Drill-down**: 각 레이어는 상위 레이어를 구체화
2. **Traceability**: 모든 코드는 Feature ID로 역추적 가능
3. **Distillation**: 코드에서 본질만 추출해 명세로 정제

---

## 예시: turbo-memory-mcp

이 프로젝트 자체가 PSDD로 문서화되어 있다:

- `docs/requirements.md` — 10개 Feature, EARS 형식
- `docs/biz-process.md` — 5개 프로세스, BPMN 드릴다운
- `docs/uis.md` — 6개 MCP Tool UI
- `docs/design.md` — TurboQuant 알고리즘, 아키텍처
- `docs/tasks.md` — 24개 완료 태스크 + 5개 백로그

각 문서를 읽으면 PSDD 적용 방법을 이해할 수 있다.
