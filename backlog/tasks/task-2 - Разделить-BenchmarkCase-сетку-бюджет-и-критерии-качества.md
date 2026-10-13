---
id: TASK-2
title: 'Разделить BenchmarkCase, сетку, бюджет и критерии качества'
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
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
- [ ] #1 Добавление PDE не требует ветвления по имени в общих компонентах
- [ ] #2 Прежние четыре train/eval конфигурации и лимиты сохранены
- [ ] #3 Проверены independent analytical residuals и boundary masks
<!-- AC:END -->
