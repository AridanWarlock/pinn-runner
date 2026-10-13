---
id: TASK-15
title: Профилировать текущий CUDA backend и проверить fused Adam
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 14:20'
labels:
  - performance
milestone: m-0
dependencies:
  - TASK-1
documentation:
  - docs/model-accuracy-plan.md
priority: medium
ordinal: 15000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ускорение не требует сначала обновлять Torch. Нужно найти долю kernel launch, derivatives и optimizer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Profiler собирается отдельно от итогового timing
- [ ] #2 Default/foreach/fused сравниваются при неизменных hyperparameters и FP32
- [ ] #3 Измерены cold wall time, warm step, память, numerical gates и resume
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
LBFGS integration exposed LightningCPUteardown (fixed,actualdeviceguard added). CorrectGPUstage2.87s vsdiagnosticCPU~1.4s for smallheat3d100calls; not acceptedhybridstrategy. Profiling dispatch/synchronizations is a candidate; no profiler proof yet.
<!-- SECTION:NOTES:END -->
