---
id: TASK-4
title: Проверить усиление веса граничных условий на CUDA
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - experiment
milestone: m-0
dependencies:
  - TASK-1
  - TASK-3
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
На позднем времени boundary RMSE примерно вдвое выше interior; исследуем, объясняет ли это основную ошибку.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Веса BC 1/3/10 сравниваются отдельно на seed 1234 и 4321 при равном бюджете
- [ ] #2 Сохранены IC/BC/interior/time/PDE метрики и wall time
- [ ] #3 Вывод различает улучшение границ и ухудшение PDE; сформулирован следующий опыт
<!-- AC:END -->
