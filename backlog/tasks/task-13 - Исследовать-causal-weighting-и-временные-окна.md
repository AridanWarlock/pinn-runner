---
id: TASK-13
title: Исследовать causal weighting и временные окна
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
updated_date: '2026-10-17 08:34'
labels:
  - conditional
milestone: m-0
dependencies:
  - TASK-1
  - TASK-6
  - TASK-8
  - TASK-10
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-tasks/task-13.md
priority: low
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Требования и критерии: docs/research-tasks/task-13.md. Сначала прочитать этот документ; подробную историю загружать только для конкретного вопроса.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 До реализации проверены temporal residuals и рост абсолютной ошибки
- [ ] #2 WholeInterval остаётся отключаемым контролем; причинный режим не маскирует early-time error
- [ ] #3 Если нужны окна, проверены continuity/checkpoint и ошибка переноса IC
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Подробности в документе требований; новые отчёты добавлять ссылками.
<!-- SECTION:NOTES:END -->
