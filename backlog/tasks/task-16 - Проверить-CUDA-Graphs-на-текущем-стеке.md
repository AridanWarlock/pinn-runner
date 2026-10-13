---
id: TASK-16
title: Проверить CUDA Graphs на текущем стеке
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - performance
milestone: m-0
dependencies:
  - TASK-7
  - TASK-15
documentation:
  - docs/model-accuracy-plan.md
priority: medium
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Маленькая MLP может ограничиваться launch overhead. Статические buffers не должны замораживать sampling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Eager/graph predict сравниваются отдельно; cold capture включён в wall time
- [ ] #2 Если training поддерживается, проверены вторые производные и parameter gradients
- [ ] #3 Неполный batch и обновление sampling сохраняют численный смысл; effective mode отражён в отчёте
<!-- AC:END -->
