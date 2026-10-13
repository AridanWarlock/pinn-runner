---
id: TASK-22
title: Выделить контракт hard constraints для heat-профиля
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 15:56'
updated_date: '2026-10-13 18:37'
labels:
  - hard-constraints
  - heat
dependencies: []
documentation:
  - docs/hard-heat-plan.md
priority: high
type: enhancement
ordinal: 22000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Существующий HardBoxInitial смешивает математическую конструкцию и исполнение. Расширение heat на другие BC требует явных гарантий, применимости и checkpoint-контракта, сохраняя прямой hard и soft как отдельные режимы.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Разрешённая стратегия сообщает точные IC/BC и оставшиеся loss; unsupported hard отклоняется без молчаливого soft fallback
- [x] #2 Eager и opt-in execution отделены от математического рецепта; горячий путь не выбирает стратегии по строкам
- [x] #3 Save/load/predict/fine-tune и прежний поддержанный hard проверены; CPU full и профильные CUDA проверки задокументированы
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Ввести immutable ConstraintPlan с exact IC/faces и remaining loss; перенести текущую валидацию из factory. 2. Отделить HardBoxInitial recipe от eager/fused/product-rule executors, выбор один раз при сборке. 3. Сохранить checkpoint identity и проверить public predict/retrain, CPU full и короткий профильный CUDA набор. Никаких изменений PDE/defaults или новой heat inference по имени функции.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
d251c49: immutable ConstraintPlan разрешает exact_initial/exact_faces/remaining_losses и source signature; validation вынесена изfactory. HardBoxInitial recipe безexecutionflags; ProductRuleHard/FusedPredictHard составлены изrecipe, dispatchодинраз. Config/defaults/checkpoint schema сохранены. FullCPU268passed101.38s,strictmypy54files,Ruff;профильные26passed. Дополнительный explicitmypy2benchmarkfiles. Новая проверка publicfine-tune product_rulecheckpoint черезeager. CUDAкороткийнабор запущен,безmatrix.

Windows d251c49 PINN_TEST_DEVICE=cuda: derivative+publicroundtrip11passed14.10s,изних6GPUderivative/4CPUdispatch+1CUDAe2e(train/refine/predict/fine-tune). Логи numerical-reports/constraint-plan-cuda.log/xml. Полнаяmatrix не повторялась; docs/constraint-plan-refactoring.md содержит scope/evidence и условность гарантии g|Γ=0 при sampled compatibility.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Выделен ConstraintPlan с гарантиями/remaining loss; HardBoxInitial recipe отделён от execution adapters, checkpoint/config/defaults сохранены. FullCPU268passed,mypy54,Ruff;профильныеCPU26passed;Windows11passed(6CUDA derivatives+1CUDApublicroundtrip,4CPU). Unsupported hard отклоняется; sampled compatibility документирована. Новые BC остаются TASK-23/24.
<!-- SECTION:FINAL_SUMMARY:END -->
