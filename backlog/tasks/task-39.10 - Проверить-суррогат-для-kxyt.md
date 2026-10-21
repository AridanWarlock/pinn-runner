---
id: TASK-39.10
title: 'Проверить суррогат для k(x,y,t)'
status: To Do
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 08:31'
labels: []
dependencies:
  - TASK-39.6
  - TASK-39.9
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-research-roadmap.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-terminology.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-validation.md
parent_task_id: TASK-39
priority: medium
ordinal: 49000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
После двумерной пространственной и одномерной пространственно-временной ступеней проверить их перенос на общий коэффициент двух координат и времени.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Определено аналитически проверяемое семейство с существенной зависимостью от x,y,t и совместимыми данными.
- [ ] #2 Опубликованы ограниченный протокол и новые функции проверки; общие веса оценены без обучения на каждой гипотезе отдельно.
- [ ] #3 В отчёте раздельно представлены точность, ошибка кода, время, память, неудачи и ограничения переноса.
<!-- AC:END -->
