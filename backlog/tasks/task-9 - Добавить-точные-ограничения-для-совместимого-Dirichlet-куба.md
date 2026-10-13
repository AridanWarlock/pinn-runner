---
id: TASK-9
title: Добавить точные ограничения для совместимого Dirichlet куба
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
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
- [ ] #1 Soft и hard режимы выбираются явно; проверены IC/BC и производные
- [ ] #2 Неподдерживаемые Neumann/Robin или несовместимые IC/transform отвергаются понятно
- [ ] #3 Checkpoint/predict/resume сохраняют смысл physical fields
<!-- AC:END -->
