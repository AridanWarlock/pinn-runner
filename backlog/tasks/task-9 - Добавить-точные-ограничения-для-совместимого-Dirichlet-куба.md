---
id: TASK-9
title: Добавить точные ограничения для совместимого Dirichlet куба
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 09:39'
labels:
  - adapter
milestone: m-0
dependencies:
  - TASK-3
documentation:
  - docs/model-accuracy-plan.md
priority: medium
ordinal: 9000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ошибка границ может доминировать. Ограничения должны использовать заданные IC/BC, без знания точного решения во времени.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Soft и hard режимы выбираются явно; проверены IC/BC и производные
- [x] #2 Неподдерживаемые Neumann/Robin или несовместимые IC/transform отвергаются понятно
- [x] #3 Checkpoint/predict/resume сохраняют смысл physical fields
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Explicit soft/hard initial homogeneous Dirichlet box constraint. Compile tensor output transform g(x)+tau B(x)N(x,t), known IC only; reject incompatible BC and existing user transform. Skip redundant IC/BC loss in hard mode; persist signature and check predict/fine-tune. Verify boundary/initial/second derivatives, run full CPU + smoke, then CUDA comparison TASK-10.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Final verification:Mac full233passed,Ruff/format/strictmypy; WindowsCUDAhardware5passed0skipped(15.49s),41component tests passed on1246868. Genuine numeric hard trial evidence in TASK10/report; no default promotion.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Soft/hard tensor transform implemented for compatible homogeneous Dirichlet boxes without teacher; skip satisfied IC/BC loss, preserve second-coordinate and parameter derivatives. Reject incompatible faces/IC and custom transforms/parameter families. Checkpoint signature guards physical meaning. Full231 tests, then fast215 (17 excluded), hard CPU 60-step train/predict smoke, derivative and checkpoint incompatibility tests. CUDA screening TASK-10 next.
<!-- SECTION:FINAL_SUMMARY:END -->
