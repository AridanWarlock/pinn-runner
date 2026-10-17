---
id: TASK-19
title: Исследовать compile и functional derivative backend
status: Deferred
assignee: []
created_date: '2026-10-13 08:37'
updated_date: '2026-10-17 08:34'
labels:
  - runtime
milestone: m-1
dependencies:
  - TASK-18
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-tasks/task-19.md
priority: low
ordinal: 19000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Требования и критерии: docs/research-tasks/task-19.md. Сначала прочитать этот документ; подробную историю загружать только для конкретного вопроса.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Predict и residual-training режимы проверены отдельно на выбранной версии
- [ ] #2 Производные, parameter gradients, graph breaks, prep time и recompilations измерены
- [ ] #3 Windows и Linux Docker backend различены; непроверенный fallback не называется compiled
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Подробности в документе требований; новые отчёты добавлять ссылками.
<!-- SECTION:NOTES:END -->
