---
id: TASK-39.3
title: Проверить аналитическое семейство k(x) с фиксированной пространственной формой
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
priority: high
ordinal: 42000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Добавить пространственную изменчивость отдельно от разнообразия форм функций. Первый кандидат: k(x)=a[1+2x(1-x)], одинаковое начальное поле 4x(1-x), источник нулевой. Изменяется масштаб a, форма пока фиксирована; это не решение для произвольной k(x).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Аналитическое решение и дивергенция потока независимо проверены; неверный оператор без производной k обнаруживается.
- [ ] #2 До обучения закреплены семейство, диапазон a, новые гипотезы и абсолютные цели; поле аналитически зависит от a при одинаковых данных.
- [ ] #3 Общая сеть оценена на новых a без переобучения; качество и стоимость сопоставлены с предыдущей ступенью, ограничение фиксированной формы явно сохранено.
<!-- AC:END -->
