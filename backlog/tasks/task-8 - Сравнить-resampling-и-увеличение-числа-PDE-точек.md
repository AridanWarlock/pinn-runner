---
id: TASK-8
title: Сравнить resampling и увеличение числа PDE-точек
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - experiment
milestone: m-0
dependencies:
  - TASK-1
  - TASK-7
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 8000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Проверяем качество holdout, не только уменьшение train residual.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fixed 1024, resampled 1024 и больший fixed budget точек оценены отдельно
- [ ] #2 Измерены independent residual, качество по времени, цена и память
- [ ] #3 Принятие опирается на seed и результат всех метрик, не удачный запуск
<!-- AC:END -->
