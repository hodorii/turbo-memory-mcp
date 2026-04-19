# Claude Project Rules

## Methodology

This project follows **PSDD (Process-augmented Spec Driven Development)**.

Read `docs/PSDD.md` at session start and follow the rules.

## Session Start Checklist

1. Read `docs/PSDD.md` — understand methodology
2. Check `docs/tasks.md` — current progress
3. Read relevant `docs/*.md` — requirements/design

## Working Rules

- All implementation based on TASK IDs in `docs/tasks.md`
- Design decisions reflected in `docs/design.md`
- New requirements added to `docs/requirements.md` with REQ IDs
- Mark `[x]` in `tasks.md` when task complete

## Document Structure

```
docs/
  PSDD.md           — Methodology definition
  requirements.md   — EARS requirements
  biz-process.md    — BPMN process drill-down
  uis.md            — UI specification
  design.md         — Technical design
  tasks.md          — Implementation checklist
```

## Memory MCP

This project uses `memory` MCP server.

- Session start: `recall("current work keywords")`
- Task complete: `remember(["decisions", "solutions"])`
