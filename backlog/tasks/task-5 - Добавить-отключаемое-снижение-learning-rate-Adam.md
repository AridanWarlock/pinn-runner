---
id: TASK-5
title: Добавить отключаемое снижение learning rate Adam
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 08:59'
labels:
  - adapter
milestone: m-0
dependencies: []
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Постоянный LR 0.001 может оставлять optimization floor. Разделяем scheduler и ускорение optimizer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ConstantAdam сохраняет прежнее поведение, decaying режим явно выбирается
- [x] #2 Scheduler считает optimizer steps; проверены границы, resume и effective LR
- [x] #3 Состояние scheduler и версия стратегии сохраняются
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Explicit constant/exponential LR settings with final_ratio, step-based schedule. 2. Store scheduler recipe and configure interval=step through OptimizationPolicy. 3. Test exact endpoint and scheduler state roundtrip. 4. Extend benchmark adapters/CLI, run CPU smoke/full. 5. TASK-6 compares decay against equal-budget and longer fixed-LR controls.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Constant/exponential RatePolicy, strict schedule parser and step-based Lightning scheduler implemented. CPU full 224 passed; real 60-step smoke checkpoint ends at LR .0001 and last_epoch=60. Optimizer/scheduler state roundtrip reproduces updates exactly. CLI checkpoint remains fine-tune, not optimizer resume; documented.
<!-- SECTION:FINAL_SUMMARY:END -->
