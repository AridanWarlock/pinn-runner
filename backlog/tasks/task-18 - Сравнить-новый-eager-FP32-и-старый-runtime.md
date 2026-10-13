---
id: TASK-18
title: Сравнить новый eager FP32 и старый runtime
status: Deferred
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - runtime
milestone: m-1
dependencies:
  - TASK-17
documentation:
  - docs/model-accuracy-plan.md
priority: low
ordinal: 18000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Само обновление может быть maintenance без ускорения; версии не смешиваем с новыми математическими стратегиями.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Изолированный old/new eager опыт при одинаковом PDE/recipe
- [ ] #2 Проверены derivative/checkpoint/resume и cold/warm performance на всех PDE
- [ ] #3 Full CPU, CUDA и целевой Docker проверены до обновления зависимостей
<!-- AC:END -->
