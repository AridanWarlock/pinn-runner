---
id: TASK-3
title: Ввести именованные loss и отключаемый LossPolicy
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 08:54'
labels:
  - adapter
milestone: m-0
dependencies: []
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
IC/BC/PDE сейчас складываются без индивидуальных весов. Нужна основа для контролируемых опытов, не изменение defaults.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 EqualWeights воспроизводит текущий loss; FixedWeights задаётся строгим YAML
- [x] #2 Регистрируются IC/BC/PDE losses и периодические gradient norms без изменения обучения
- [x] #3 Resolved strategy set сохраняется в отчёте и checkpoint; отключение возвращает контроль
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Strict training.loss parser equal/fixed + positive IC/BC/PDE weights, diagnostic interval. 2. LossPolicy combines named scalar terms; log components and optional autograd.grad norms without touching optimizer gradients. 3. Persist recipe in resolved config/checkpoint/report. 4. CPU equivalence and diagnostic non-mutation tests, full/e2e. 5. Publish and run CUDA weight ablation in TASK-4.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Equal/fixed LossPolicy, strict YAML, named losses and optional read-only gradient norms implemented. Full CPU 216 passed; subsequent fast 200 passed (17 excluded), CPU weighted smoke and actual checkpoint/report/CSV diagnostics verified. Metadata persisted. New recipe controls are explicit; no default promotion.
<!-- SECTION:FINAL_SUMMARY:END -->
