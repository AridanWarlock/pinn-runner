---
id: TASK-39.6
title: 'Перенести суррогат теплопроводности на k(x,y)'
status: To Do
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 08:31'
labels: []
dependencies:
  - TASK-39.5
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-research-roadmap.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-terminology.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-validation.md
parent_task_id: TASK-39
priority: medium
ordinal: 45000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
После подтверждения одномерной формы исследовать скалярный изотропный коэффициент двух пространственных координат на освоенной геометрии. Не подменять его тензором или фактически одномерным примером.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Заданы аналитические случаи с существенной зависимостью k от x и y, совместимыми данными и независимыми контролями потока.
- [ ] #2 До обучения опубликована ограниченная матрица; одна общая модель оценена на новых двумерных коэффициентах без переобучения.
- [ ] #3 Раздельно подтверждены качество, стоимость, ошибка кодирования и сохранённый файловый контракт; ограничения и неудачи отражены в отчёте.
<!-- AC:END -->
