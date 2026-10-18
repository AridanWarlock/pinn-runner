---
id: TASK-16
title: Проверить CUDA Graphs на текущем стеке
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
updated_date: '2026-10-18 11:30'
labels:
  - performance
milestone: m-0
dependencies:
  - TASK-7
  - TASK-15
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-tasks/task-16.md
priority: low
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Требования и критерии: docs/research-tasks/task-16.md. Сначала прочитать этот документ; подробную историю загружать только для конкретного вопроса.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Eager/graph predict сравниваются отдельно; cold capture включён в wall time
- [ ] #2 Если training поддерживается, проверены вторые производные и parameter gradients
- [ ] #3 Неполный batch и обновление sampling сохраняют численный смысл; effective mode отражён в отчёте
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Подробности в документе требований; новые отчёты добавлять ссылками.
<!-- SECTION:NOTES:END -->
