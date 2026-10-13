---
id: TASK-8
title: Сравнить resampling и увеличение числа PDE-точек
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 09:27'
labels:
  - experiment
milestone: m-0
dependencies:
  - TASK-1
  - TASK-7
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 8000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Проверяем качество holdout, не только уменьшение train residual.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fixed 1024, resampled 1024 и больший fixed budget точек оценены отдельно
- [x] #2 Измерены independent residual, качество по времени, цена и память
- [x] #3 Принятие опирается на seed и результат всех метрик, не удачный запуск
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Control fixed1024 from bc-weight-1-001; fresh1024 every epoch versus fixed4096 at same6000 steps/2seeds/BC1/constantLR/grad_every200. Compare independent residual, all solution metrics, time; note CombinedLoader epoch cadence changes when dataset size changes.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Two-seed heat3d screening at6000: random refresh1024 not robust; fixed4096 lowers PDE/final errors but global L2 remains5.22/5.30%. Opt-in only. All time/error metrics collected; separate256-step allocator calibration: same18.47MB allocated/23.07MB reserved for fixed/resampled/dense, not a full6000-step memory measurement.
<!-- SECTION:FINAL_SUMMARY:END -->
