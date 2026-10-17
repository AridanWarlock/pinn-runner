---
id: TASK-18
title: Сравнить новый eager FP32 и старый runtime
status: Deferred
assignee: []
created_date: '2026-10-13 08:37'
updated_date: '2026-10-17 08:34'
labels:
  - runtime
milestone: m-1
dependencies:
  - TASK-17
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-tasks/task-18.md
priority: low
ordinal: 18000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Требования и критерии: docs/research-tasks/task-18.md. Сначала прочитать этот документ; подробную историю загружать только для конкретного вопроса.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Изолированный old/new eager опыт при одинаковом PDE/recipe
- [ ] #2 Проверены derivative/checkpoint/resume и cold/warm performance на всех PDE
- [ ] #3 Full CPU, CUDA и целевой Docker проверены до обновления зависимостей
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Подробности в документе требований; новые отчёты добавлять ссылками.
<!-- SECTION:NOTES:END -->
