---
id: TASK-39.8
title: Проверить аналитический суррогат для k(t) как мост к переменному времени
status: To Do
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 08:31'
labels: []
dependencies:
  - TASK-39.2
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-research-roadmap.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-terminology.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-validation.md
parent_task_id: TASK-39
priority: medium
ordinal: 47000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Отделить временную изменчивость от пространственной: для k(t) доступно аналитическое решение через интеграл коэффициента. Это дополнительная простая ступень, не замена k(x,t).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Задано положительное семейство k(t) с аналитическим интегралом и одинаковыми начальными/граничными данными; решения различаются для разных гипотез.
- [ ] #2 До обучения фиксированы новые временные функции, бюджет, абсолютные цели и окончание; общая модель проверена на новых k(t).
- [ ] #3 Проверено сохранение целой временной функции на входе, качества и стоимости; отсутствие пространственной изменчивости указано явно.
<!-- AC:END -->
