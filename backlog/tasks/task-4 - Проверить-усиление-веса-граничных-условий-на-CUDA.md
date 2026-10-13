---
id: TASK-4
title: Проверить усиление веса граничных условий на CUDA
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 09:01'
labels:
  - experiment
milestone: m-0
dependencies:
  - TASK-1
  - TASK-3
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
На позднем времени boundary RMSE примерно вдвое выше interior; исследуем, объясняет ли это основную ошибку.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Веса BC 1/3/10 сравниваются отдельно на seed 1234 и 4321 при равном бюджете
- [x] #2 Сохранены IC/BC/interior/time/PDE метрики и wall time
- [x] #3 Вывод различает улучшение границ и ухудшение PDE; сформулирован следующий опыт
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Compare BC weights 1/3/10, 6000 steps, seeds 1234/4321, gradient_norm_every=200 for all. Analyze total/final/interior/BC/PDE and time, do not promote defaults from 2-seed screening.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Screening BC 1/3/10 complete on heat3d, 6000 steps, 2 seeds, same grad diagnostic cadence. BC3 global L2 .06470/.05686, BC10 second seed .07170: no robust gain vs .04682/.05756. Do not promote weights. Detailed report paths numerical-reports/bc-weight-{1,3,10}-001/report.json; investigate loss conflict rather than increase BC weight blindly.
<!-- SECTION:FINAL_SUMMARY:END -->
