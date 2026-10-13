---
id: TASK-5
title: Добавить отключаемое снижение learning rate Adam
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - adapter
milestone: m-0
dependencies: []
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Постоянный LR 0.001 может оставлять optimization floor. Разделяем scheduler и ускорение optimizer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ConstantAdam сохраняет прежнее поведение, decaying режим явно выбирается
- [ ] #2 Scheduler считает optimizer steps; проверены границы, resume и effective LR
- [ ] #3 Состояние scheduler и версия стратегии сохраняются
<!-- AC:END -->
