---
id: TASK-20
title: Оценить TF32 и mixed precision отдельно от прозрачных ускорений
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
ordinal: 20000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Низкая точность может портить second derivatives; бизнес-смысл проверяется численно.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Strict FP32 остаётся контролем; TF32/FP16/BF16 раздельные опыты
- [ ] #2 Проверены derivative accuracy относительно FP64 и solution/residual gates
- [ ] #3 Режимы не включаются по номеру библиотеки или общему fast переключателю
<!-- AC:END -->
