---
id: TASK-6
title: Проверить scheduler и увеличение бюджета как разные гипотезы
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - experiment
milestone: m-0
dependencies:
  - TASK-1
  - TASK-5
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Нужно отличить влияние меньшего LR от простого увеличения количества шагов.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Контроль, decay и более длинный fixed-LR запуск сравниваются раздельно
- [ ] #2 Два seed, фиксированные evaluation/holdout, отчёт качества и времени
- [ ] #3 Вывод объясняет, есть ли optimization floor и стоит ли расширять опыт
<!-- AC:END -->
