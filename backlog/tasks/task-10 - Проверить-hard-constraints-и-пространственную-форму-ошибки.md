---
id: TASK-10
title: Проверить hard constraints и пространственную форму ошибки
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 09:39'
labels:
  - experiment
milestone: m-0
dependencies:
  - TASK-1
  - TASK-9
documentation:
  - docs/model-accuracy-plan.md
priority: medium
ordinal: 10000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Точная граница должна подтвердить либо опровергнуть гипотезу о boundary defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Soft и hard режимы проверены на одинаковых seed и бюджетах
- [x] #2 Измерены interior и spatial-mode errors, не только исчезновение BC error
- [x] #3 Проверены применимые PDE и цена; результат не обобщается на все границы
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Compare hard_initial_dirichlet6000 two seeds versus soft control6000, same PDE/eval and fixed LHS1024. Inspect exact IC/BC, interior errors, independent residual, speed and peak memory. Keep opt-in pending 5 seeds/all applicable PDE. Add short256-step allocator calibration for soft/resampled/dense, clearly separate from long quality run.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Additional matched3000-step heat3d budget completed. Burgers seed4321 doubled6000 to test amplitude convergence; extending heat3d confirmation to predeclared seeds19/2026/73, paired soft controls. This does not satisfy TASK14 all-PDE/composition/default acceptance.

Matched heat3d3000:soft global9.16/11.39%,final22.17/26.55%,29.7s;hard global0.338/0.357%,final0.889/1.227%,40s. Hard cheaper to achieve quality than6000/12000soft despite higher per-step cost. Burgers4321 at6000:hard global0.056%,final0.138%,PDE0.00277 vssoft0.192%,0.310%,0.00526; supports budget-limited amplitude convergence.

Five predeclared heat3d seeds completed:hard improved global/final L2 in every pair;median global5.596%->0.316%,final13.711%->1.475%;worst hard0.432%/1.632%. All5hard quality gates pass,2/5soft pass. Median training56.32->76.58s;warm inference19.43->46.87ms. Report docs/accuracy-experiments-2026-10-13.md and compactCSV/plot.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
36long trials overall192000steps +3memory calibrations. Hard heat3d improved all5predeclared seeds:median global0.316% and final1.475% vssoft5.596%/13.711%; allhard gates pass. Shape/interior verified; heat2d benefits, Burgers4321 needs6000budget. Trainingmedian+36%,warm inference2.41x,hard3000 better/cheaper thanlongsoft for first2seeds. Detailed report/CSV/plot committed on model-accuracy; general defaults remainpendingTASK14.
<!-- SECTION:FINAL_SUMMARY:END -->
