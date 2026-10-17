---
id: TASK-32
title: Завершить передачу через скиллы и удалить устаревшие точки входа
status: Done
assignee:
  - '@codex'
created_date: '2026-10-17 15:30'
updated_date: '2026-10-17 15:32'
labels: []
dependencies:
  - TASK-31
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/agent-start.md
priority: high
type: docs
ordinal: 32000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Перед архивированием чата нужны два коротких проектных скилла и одна актуальная точка входа. Устаревшие экспорты противоречат текущим статусам; математические выводы и научные результаты должны сохраниться.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Два проектных скилла валидны и ведут к актуальным инструментам и математическим требованиям.
- [x] #2 Устаревшие стартовые экспорты удалены, ссылки исправлены; математические объяснения сохранены.
- [x] #3 Проверены ссылки и инструкции, передача готова без загрузки всей истории.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Создать два компактных SKILL.md. 2. Удалить старые стартовые экспорты и дублирующую памятку, сохранив математические материалы. 3. Обновить вход и ссылки, проверить и опубликовать.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Добавлены pinn-research и pinn-gpu-experiments; quick_validate прошёл для обоих. Удалены три staged-экспорта и дублирующая памятка. Учебник переименован в math-tutor.md; основная математическая часть побайтно сохранена. Проверены 17 ссылок и отсутствие старых имён; offline plan работает. Python не менялся, общие тесты не повторялись. Коммит a9445c7.
<!-- SECTION:FINAL_SUMMARY:END -->
