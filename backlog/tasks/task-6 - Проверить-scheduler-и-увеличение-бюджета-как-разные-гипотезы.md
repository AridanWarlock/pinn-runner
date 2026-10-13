---
id: TASK-6
title: Проверить scheduler и увеличение бюджета как разные гипотезы
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 09:14'
labels:
  - experiment
milestone: m-0
dependencies:
  - TASK-1
  - TASK-5
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Нужно отличить влияние меньшего LR от простого увеличения количества шагов.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Контроль, decay и более длинный fixed-LR запуск сравниваются раздельно
- [x] #2 Два seed, фиксированные evaluation/holdout, отчёт качества и времени
- [x] #3 Вывод объясняет, есть ли optimization floor и стоит ли расширять опыт
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Compare constant 6000 baseline bc-weight-1-001 versus decay final_ratio=.1 at 6000 and constant 12000, seeds1234/4321, BC1 and grad norms every200. Keep profiles otherwise equal, compare solution/time/independent residual and price.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
6000-step early exponential decay .1 worsens both seeds (L2 .09045/.11217); constant12000 passes global quality. Compare reports lr-decay-001/lr-long-001 against bc-weight-1-001. Reject this decay recipe as default; cannot reject all annealing. Hypothesis: decay too early before learning spatial shape; test delayed decay later. Full temporal/residual/time reports collected.
<!-- SECTION:FINAL_SUMMARY:END -->
