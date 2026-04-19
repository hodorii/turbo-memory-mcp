# Gemini CLI Project Rules

## 방법론

이 프로젝트는 **PSDD (Process-augmented Spec Driven Development)** 방법론을 따릅니다.

세션 시작 시 반드시 `docs/PSDD.md`를 읽고 규칙을 따르세요.

## 세션 시작 체크리스트

1. `docs/PSDD.md` 읽기 — 방법론 이해
2. `docs/tasks.md` 확인 — 현재 진행 상태 파악
3. 관련 `docs/*.md` 읽기 — 요구사항/설계 파악

## 작업 규칙

- 모든 구현은 `docs/tasks.md`의 TASK ID 기준
- 설계 결정은 `docs/design.md`에 반영
- 새 요구사항은 `docs/requirements.md`에 REQ ID로 추가
- 작업 완료 시 `tasks.md`에서 `[x]` 체크

## 문서 구조

```
docs/
  PSDD.md           — 방법론 정의
  requirements.md   — EARS 요구사항
  biz-process.md    — BPMN 프로세스
  uis.md            — UI 명세
  design.md         — 기술 설계
  tasks.md          — 체크리스트
```

## Memory MCP

이 프로젝트는 `memory` MCP 서버를 사용합니다.

- 세션 시작: `recall("현재 작업 키워드")`
- 작업 완료: `remember(["결정사항", "해결책"])`
