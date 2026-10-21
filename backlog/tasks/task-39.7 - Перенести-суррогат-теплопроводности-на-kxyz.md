---
id: TASK-39.7
title: 'Перенести суррогат теплопроводности на k(x,y,z)'
status: To Do
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 08:31'
labels: []
dependencies:
  - TASK-39.6
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-research-roadmap.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-terminology.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-validation.md
parent_task_id: TASK-39
priority: medium
ordinal: 46000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Переносить уже подтверждённый подход в три пространственных измерения, отдельно учитывая рост стоимости и разрешение входной функции.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Подготовлены аналитические случаи с существенной зависимостью от трёх координат; источники и условия не скрывают игнорирование коэффициента.
- [ ] #2 До обучения закреплены ограниченная матрица, новые гипотезы, цели и окончание; общая модель оценена без переобучения между материалами.
- [ ] #3 Сохранены качество, стоимость, память, разрешение кода, файловая воспроизводимость и решение о границах применимости.
<!-- AC:END -->
