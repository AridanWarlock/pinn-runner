---
id: TASK-15
title: Профилировать текущий CUDA backend и проверить fused Adam
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
updated_date: '2026-10-17 08:34'
labels:
  - performance
milestone: m-0
dependencies:
  - TASK-1
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-tasks/task-15.md
priority: medium
ordinal: 15000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Требования и критерии: docs/research-tasks/task-15.md. Сначала прочитать этот документ; подробную историю загружать только для конкретного вопроса.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Profiler собирается отдельно от итогового timing
- [ ] #2 Default/foreach/fused сравниваются при неизменных hyperparameters и FP32
- [ ] #3 Измерены cold wall time, warm step, память, numerical gates и resume
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
История сохранена в docs/research-history/task-15.md.
<!-- SECTION:NOTES:END -->
