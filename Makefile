PROJ    := $(shell pwd)
PYTHON  := $(PROJ)/.venv/bin/python
PIP     := $(PROJ)/.venv/bin/pip

KIRO_MCP        := $(HOME)/.kiro/settings/mcp.json
GEMINI_MCP      := $(HOME)/.gemini/settings.json
ANTIGRAVITY_MCP := $(HOME)/.gemini/antigravity/mcp_config.json
GEMINI_CTX      := $(HOME)/.gemini/gemini.md
ANTIGRAVITY_CTX := $(HOME)/.gemini/antigravity/scratch/antigravity_agent_prompt.md

.PHONY: all install register unregister ingest test clean

all: install register

# ── 설치 ──────────────────────────────────────────────────────────────────────
install:
	python3 -m venv .venv
	$(PIP) install --quiet sentence-transformers numpy
	@echo "✓ 설치 완료"

# ── MCP 등록 (jq로 JSON 병합) ─────────────────────────────────────────────────
register: _register_kiro _register_gemini _register_antigravity _register_skills
	@echo "✓ 등록 완료 (kiro / gemini / antigravity)"

_register_kiro:
	@jq '.mcpServers.memory = {"command":"$(PYTHON)","args":["$(PROJ)/server.py"]}' \
		$(KIRO_MCP) > /tmp/_mcp.json && mv /tmp/_mcp.json $(KIRO_MCP)

_register_gemini:
	@jq '.mcpServers.memory = {"command":"$(PYTHON)","args":["$(PROJ)/server.py"]}' \
		$(GEMINI_MCP) > /tmp/_mcp.json && mv /tmp/_mcp.json $(GEMINI_MCP)

_register_antigravity:
	@jq '.mcpServers.memory = {"command":"$(PYTHON)","args":["$(PROJ)/server.py"]}' \
		$(ANTIGRAVITY_MCP) > /tmp/_mcp.json && mv /tmp/_mcp.json $(ANTIGRAVITY_MCP)

_register_skills:
	@mkdir -p $(PROJ)/.kiro/steering
	@$(PYTHON) -c "\
import os; \
skill = '\n\n## Memory MCP 사용 원칙\n\`memory\` MCP 서버가 등록되어 있다. 중요한 정보는 반드시 저장하라.\n- 저장: \`remember(texts=[...])\` — 배열 배치 우선\n- 검색: \`recall(query)\` — 세션 시작 시 관련 기억 먼저 확인\n'; \
[open(f,'a').write(skill) for f in ['$(GEMINI_CTX)','$(ANTIGRAVITY_CTX)'] if 'Memory MCP' not in open(f).read()]; \
print('✓ skills 등록 완료')"

# ── MCP 등록 해제 ─────────────────────────────────────────────────────────────
unregister:
	@for f in $(KIRO_MCP) $(GEMINI_MCP) $(ANTIGRAVITY_MCP); do \
		jq 'del(.mcpServers.memory)' $$f > /tmp/_mcp.json && mv /tmp/_mcp.json $$f; \
	done
	@echo "✓ 등록 해제 완료"

# ── 배치 ingest ───────────────────────────────────────────────────────────────
ingest:
	@test -n "$(FILE)" || (echo "사용법: make ingest FILE=memories.json" && exit 1)
	$(PYTHON) ingest.py $(FILE)

# ── 동작 테스트 ───────────────────────────────────────────────────────────────
test:
	@echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' \
		> /tmp/_test_input.json
	@echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"memory_stats","arguments":{}}}' \
		>> /tmp/_test_input.json
	@cat /tmp/_test_input.json | $(PYTHON) server.py 2>/dev/null | \
		python3 -c "import sys,json; [print(json.dumps(json.loads(l),ensure_ascii=False,indent=2)) for l in sys.stdin]"

# ── 정리 ──────────────────────────────────────────────────────────────────────
clean:
	rm -rf .venv __pycache__ memory.pkl
