---
id: TASK-31
title: Показывать текущие задачи кратким списком со ссылками на требования
status: Done
assignee:
  - '@codex'
created_date: '2026-10-17 15:07'
updated_date: '2026-10-17 15:08'
labels: []
dependencies:
  - TASK-30
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-task-index.md
priority: medium
type: enhancement
ordinal: 31000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Новая сессия должна найти актуальное поручение без чтения историй карточек и перехода к завершённой TASK-27. Требования: docs/research-task-index.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Команда tasks возвращает ограниченную страницу текущих задач, статусы и ссылки на требования без истории.
- [x] #2 Пагинация и фильтр статуса не теряют задачи; готовность доски не трактуется как разрешение менять модель.
- [x] #3 Ошибки контракта и неверные аргументы отвергаются; профильные и общие проверки пройдены, стартовый документ обновлён.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Адаптер страниц Backlog JSON и обогащение краткими карточками. 2. CLI tasks с ограничением страницы и фильтром статуса. 3. Проверки пагинации/ошибок, документация и новая точка входа.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Команда tasks показывает ограниченные страницы со ссылками на требования, без истории. Реальная доска: 7 задач на 4 страницах без повторов. Профильные 48 тестов; check.py: 733 прошли, 50 e2e исключены. Коммит fb54555. Стартовый документ больше не направляет к завершённой TASK-27.
<!-- SECTION:FINAL_SUMMARY:END -->
