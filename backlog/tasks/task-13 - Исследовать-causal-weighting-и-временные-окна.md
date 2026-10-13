---
id: TASK-13
title: Исследовать causal weighting и временные окна
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - conditional
milestone: m-0
dependencies:
  - TASK-1
  - TASK-6
  - TASK-8
  - TASK-10
documentation:
  - docs/model-accuracy-plan.md
priority: low
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Текущие данные показывают главным образом decay и spatial defect. Причинность пока не доказана.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 До реализации проверены temporal residuals и рост абсолютной ошибки
- [ ] #2 WholeInterval остаётся отключаемым контролем; причинный режим не маскирует early-time error
- [ ] #3 Если нужны окна, проверены continuity/checkpoint и ошибка переноса IC
<!-- AC:END -->
