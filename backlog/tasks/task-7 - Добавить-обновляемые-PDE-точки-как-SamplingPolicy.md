---
id: TASK-7
title: Добавить обновляемые PDE-точки как SamplingPolicy
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 09:05'
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
- [x] #1 FixedLHS и ResampledLHS переключаются настройкой без изменения IC/BC
- [x] #2 Seed, счётчик обновлений и sampling state проверены для resume
- [x] #3 Частота обновлений документирована и покрыта CPU тестами
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use deterministic epoch-indexed LHS refresh as SamplingPolicy, only equation sampler. Preserve initial fixed dataset for control, refresh at configured epochs, save current collocation tensors/counter/seed in datamodule state; validate state roundtrip and CPU runner. Fixed mode unchanged. Then TASK-8 compares refresh versus more fixed points.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed/resampled LHS selectable. Only equation points refresh; independent RNG and saved datamodule tensors/signature/epoch. Full 226 tests passed; after prefetch fix fast209 passed (17 excluded), real 60-step CPU runner checkpoint contains refreshed epoch7. LHS strata/RNG isolation/state roundtrip tested. Refresh moved before dataloader prefetch, not epoch_start hook.
<!-- SECTION:FINAL_SUMMARY:END -->
