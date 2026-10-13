---
id: TASK-1
title: Довести временную диагностику и независимый PDE holdout
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - diagnostics
milestone: m-0
dependencies: []
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Общий L2 скрывает ошибки поздних времён и границ. Основные time metrics уже добавлены в 97313b6; требуется завершить диагностику перед экспериментами.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Отчёт содержит PDE residual по фиксированным временным bins и worst-time L2 с корректным нулевым знаменателем
- [ ] #2 Границы и interior оцениваются раздельно; диагностика не изменяет параметры модели
- [ ] #3 Результаты heat3d-005 воспроизводятся из checkpoint без нового обучения
<!-- AC:END -->
