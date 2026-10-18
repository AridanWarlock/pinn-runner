---
id: TASK-14
title: Подтвердить комбинацию стратегий и обновить standard defaults
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
updated_date: '2026-10-18 11:30'
labels:
  - acceptance
milestone: m-0
dependencies:
  - TASK-4
  - TASK-6
  - TASK-8
  - TASK-10
  - TASK-11
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-tasks/task-14.md
priority: low
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Требования и критерии: docs/research-tasks/task-14.md. Сначала прочитать этот документ; подробную историю загружать только для конкретного вопроса.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Пять seed 1234/4321/19/2026/73 на heat2d/heat3d/advection/Burgers
- [ ] #2 Проверены общий/final/worst-time L2, IC/BC/PDE, скорость и память
- [ ] #3 Выполнены leave-one-out абляции; resolved standard версионирован и имеет control профиль
- [ ] #4 Full CPU, Windows CUDA и численные gates пройдены до принятия defaults
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
История сохранена в docs/research-history/task-14.md.
<!-- SECTION:NOTES:END -->
