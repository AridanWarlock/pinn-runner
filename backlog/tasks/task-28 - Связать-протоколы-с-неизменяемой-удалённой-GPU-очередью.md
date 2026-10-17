---
id: TASK-28
title: Связать протоколы с неизменяемой удалённой GPU-очередью
status: To Do
assignee: []
created_date: '2026-10-17 08:29'
updated_date: '2026-10-17 08:34'
labels: []
dependencies:
  - TASK-27
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-workflow-api.md
priority: high
type: feature
ordinal: 28000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Текущий сборщик проверяет локальные отчёты; общий запуск серий всё ещё требует временных SSH-рецептов. Цель и требования: docs/research-workflow-api.md, раздел следующего этапа.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Протокол и опубликованный SHA закреплены до старта, повторный ID не запускает дубликат, используется единая GPU-блокировка.
- [ ] #2 Статус, обрыв SSH, таймаут и дерево процессов обработаны явно; все Windows-операции остаются в обязательном каталоге.
- [ ] #3 Удалённый сбор возвращает компактные проверенные данные; манифест фиксирует ожидаемые грани и происхождение исходников.
- [ ] #4 Поддельный транспорт покрывает ошибки на Mac; короткая Windows-проверка не требует длительного обучения.
<!-- AC:END -->
