---
id: TASK-28
title: Связать протоколы с неизменяемой удалённой GPU-очередью
status: Done
assignee:
  - '@codex'
created_date: '2026-10-17 08:29'
updated_date: '2026-10-17 13:16'
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
- [x] #1 Протокол и опубликованный SHA закреплены до старта, повторный ID не запускает дубликат, используется единая GPU-блокировка.
- [x] #2 Статус, обрыв SSH, таймаут и дерево процессов обработаны явно; все Windows-операции остаются в обязательном каталоге.
- [x] #3 Удалённый сбор возвращает компактные проверенные данные; манифест фиксирует ожидаемые грани и происхождение исходников.
- [x] #4 Поддельный транспорт покрывает ошибки на Mac; короткая Windows-проверка не требует длительного обучения.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Расширить существующий worker на опубликованный протокол сходимости и неизменяемый снимок. 2. Идемпотентная подготовка, отсоединённый запуск, статус/отмена и единая блокировка. 3. Компактный удалённый сбор с ожидаемыми гранями. 4. Поддельный транспорт, полный CPU и короткая проверка Windows; сохранить ограничения.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Код исполнения7cbc0d1, протокол ee71ae4, итог4a0a13f в codex/research-tools. Требования и ограничения: docs/research-workflow-api.md. Результаты и передача: docs/research-queue-result-2026-10-17.md; docs/agent-start.md.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Реализованы опубликованный протокол/снимок, общий GPU-замок, отсоединённый процесс, статус/отмена и удалённый компактный сбор с геометрией/архитектурой/точностью. CPU full738, заключительный fast688 (50e2e исключены), профиль56. На Windows подтверждены переживание SSH, таймаут, отмена, повторный/конфликтный ID и сбор2 минимальных CUDA-обучений. WMI отвергнут по правам и заменён обычным CreateProcess; права не менялись. Качество PDE минимальным опытом не подтверждено; defaults сохранены.
<!-- SECTION:FINAL_SUMMARY:END -->
