---
id: TASK-2
title: 'Разделить BenchmarkCase, сетку, бюджет и критерии качества'
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 09:36'
labels:
  - architecture
milestone: m-0
dependencies: []
documentation:
  - docs/model-accuracy-plan.md
priority: medium
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Выбор решения и границ уже полиморфный; сетки, бюджеты и quality gates остаются зашиты в config и metrics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Добавление PDE не требует ветвления по имени в общих компонентах
- [x] #2 Прежние четыре train/eval конфигурации и лимиты сохранены
- [x] #3 Проверены independent analytical residuals и boundary masks
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Extract GridPolicy, Budget, QualityPolicy from BenchmarkCase config. Explicit per-case grid data removes dimension/name special cases. Variants inherit quality metadata. Snapshot compare original eight configs and analytic gates, then fast/full targeted checks. Keep numerical recipes identical.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Final custom512-point regression exposed defaultCollocationRefresh1024 overwrite; fixed optional override preserving case budget. Full check repeated after correction:233passed82.93s, Ruff, format, strict mypy.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
GridPolicy, TrainingBudget and QualityPolicy extracted; no PDE name switches in common configuration. Eight original configs compared exactly; analytic and custom-policy tests passed. Full check: Ruff, format, strict mypy, 233 tests.
<!-- SECTION:FINAL_SUMMARY:END -->
