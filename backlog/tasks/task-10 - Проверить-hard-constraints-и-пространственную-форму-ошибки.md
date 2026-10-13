---
id: TASK-10
title: Проверить hard constraints и пространственную форму ошибки
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
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
- [ ] #1 Soft и hard режимы проверены на одинаковых seed и бюджетах
- [ ] #2 Измерены interior и spatial-mode errors, не только исчезновение BC error
- [ ] #3 Проверены применимые PDE и цена; результат не обобщается на все границы
<!-- AC:END -->
