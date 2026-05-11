# Biz-Process: Second Brain (Memory & Retrieval) Interaction

## 1. Process Overview
사용자가 지식을 프로젝트/도메인 단위로 조직하고, 상황(세션)에 맞는 기억만 선택적으로 불러오는 흐름.

## 2. Task Flow

### Task 1: Scoping (작업 영역 설정)
*   **FunctionGroup/UI**: `turboquant scope [NAME]`
*   **Step: 문맥 활성화**
    *   Detail: 프로젝트 경로 확인 및 활성 프로젝트 스택(Default Scope) 전환
    *   Logic: `SessionManager.switch_project(name); SessionManager.set_implicit_context(name)`
*   **Step: 가시성 확보**
    *   Detail: CLI 프롬프트 업데이트 (사용자에게 현재 범위 명시)
    *   Logic: `CLI.set_prompt_decorator(f"[{name}] > ")`

### Task 2: Implicit Ingestion (자동 맥락 저장)
*   **FunctionGroup/UI**: `turboquant remember "[내용]"`
*   **Step: 현재 컨텍스트 바인딩**
    *   Detail: 마지막으로 사용된 `project_id` 자동 바인딩
    *   Logic: `scope = SessionManager.get_implicit_context(); entry.project_id = scope.id`
*   **Step: 압축 및 지식 레이어 저장**
    *   Detail: 임베딩 -> 양자화(Eden) -> 메타데이터와 함께 저장
    *   Logic: `MemoryStore.add(text, embedding, project_id=scope.id)`

### Task 3: Focused Retrieval (맥락 중심 검색)
*   **FunctionGroup/UI**: `turboquant recall "[질문]"`
*   **Step: 스코프 기반 검색범위 한정 (RAG Scoping)**
    *   Detail: 현재 컨텍스트(Project/Session)에 속한 기억으로 검색 후보군 제한
    *   Logic: `SELECT * FROM entries WHERE project_id = ? AND (session_id = ? OR session_id IS NULL)`
*   **Step: 하이브리드 지식 추출**
    *   Detail: 필터링된 후보군 내 벡터 유사도 + FTS5 가중치 결합
    *   Logic: `hybrid_score = (vector_sim * 0.3) + (fts_score * 0.7)`

### Task 4: Knowledge Refinement (지식 재조직)
*   **FunctionGroup/UI**: `turboquant promote --entry_id [ID] --to_project [NEW_PROJECT]`
*   **Step: 지식 관계(Edge) 변경**
    *   Detail: 기존 엔트리의 `project_id`를 새로운 프로젝트로 마이그레이션
    *   Logic: `UPDATE entries SET project_id = ? WHERE id = ?`
