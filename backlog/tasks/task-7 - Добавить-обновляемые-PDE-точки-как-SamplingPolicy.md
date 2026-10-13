---
id: TASK-7
title: Добавить обновляемые PDE-точки как SamplingPolicy
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - adapter
milestone: m-0
dependencies: []
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
1024 фиксированные LHS точки в 4D могут давать переобучение residual. Изменение точек должно быть воспроизводимым.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 FixedLHS и ResampledLHS переключаются настройкой без изменения IC/BC
- [ ] #2 Seed, счётчик обновлений и sampling state проверены для resume
- [ ] #3 Частота обновлений документирована и покрыта CPU тестами
<!-- AC:END -->
