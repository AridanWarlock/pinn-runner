---
id: TASK-19
title: Исследовать compile и functional derivative backend
status: Deferred
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - runtime
milestone: m-1
dependencies:
  - TASK-18
documentation:
  - docs/model-accuracy-plan.md
priority: low
ordinal: 19000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Компиляция forward не доказывает поддержку PINN derivative chain. Windows Triton/toolchain — отдельный риск.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Predict и residual-training режимы проверены отдельно на выбранной версии
- [ ] #2 Производные, parameter gradients, graph breaks, prep time и recompilations измерены
- [ ] #3 Windows и Linux Docker backend различены; непроверенный fallback не называется compiled
<!-- AC:END -->
