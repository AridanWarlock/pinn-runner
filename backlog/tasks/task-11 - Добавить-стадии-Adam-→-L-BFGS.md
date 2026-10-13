---
id: TASK-11
title: Добавить стадии Adam → L-BFGS
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - adapter
milestone: m-0
dependencies:
  - TASK-3
  - TASK-5
  - TASK-7
documentation:
  - docs/model-accuracy-plan.md
priority: medium
ordinal: 11000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Улучшение оптимизации требует фиксированного objective в closure и честного учёта вычислений.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Есть явное переключение constant Adam / staged optimizer
- [ ] #2 Closure не меняет точки между evaluation; сохранены состояния стадий
- [ ] #3 Считаются closure/residual evaluations и сравнение при равном wall budget
<!-- AC:END -->
