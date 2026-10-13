---
id: TASK-14
title: Подтвердить комбинацию стратегий и обновить standard defaults
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 09:38'
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
  - docs/model-accuracy-plan.md
priority: high
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Отдельные улучшения могут конфликтовать. В default включаем только выигрыш подтверждённой композиции.
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
TASK10 now includes five-seed paired heat3d hard/soft validation. This is partial evidence only:all-PDE5seed composition/leave-one-out and Adam->LBFGS dependency remain pending. Do not mark AC1/AC3 or change defaults.
<!-- SECTION:NOTES:END -->
