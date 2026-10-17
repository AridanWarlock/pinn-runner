---
id: TASK-29
title: Добавить компактную диагностику и целостность результатов очереди
status: Done
assignee:
  - '@codex'
created_date: '2026-10-17 13:19'
updated_date: '2026-10-17 13:28'
labels: []
dependencies:
  - TASK-28
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-diagnostics.md
priority: high
type: enhancement
ordinal: 29000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Следующий агент должен различать отказ запуска, неопределённого владельца и повреждённый результат без чтения больших журналов. Требования: docs/research-diagnostics.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Локальная и удалённая диагностика дают ограниченный вывод, неизвестные/повреждённые метаданные не считаются успехом, чтение журналов ограничено до загрузки.
- [x] #2 Однозначный отказ создания процесса сохраняет failed; неоднозначность после запуска не вызывает повторной копии.
- [x] #3 Сравнение принимает только полностью записанный пакет с проверенными файлами; повреждение и частичная запись обнаруживаются.
- [x] #4 Профильные и необходимые общие проверки пройдены; короткая Windows-проверка использует существующие результаты без нового обучения.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Ограниченная диагностика и устойчивый статус. 2. Явное завершение отказа запуска. 3. Маркер завершения и контрольные суммы пакета. 4. Тесты с повреждениями/ошибками записи и реальное чтение Windows; обновить передачу.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Код057ec98, итог0ec575b в codex/research-tools. Требования: docs/research-diagnostics.md. Доказательства: docs/research-diagnostics-result-2026-10-17.md; передача docs/agent-start.md.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Добавлены ограниченная локальная/удалённая диагностика, failed при однозначном отказе запуска и завершающий bundle.json с проверкой JSON/CSV. Профиль74, полный755, финальный fast706 (50e2e исключены). На Windows прочитаны существующие q28-probe/q28-tiny; хвост512 из4214байтов; пакет пересобран, научные данные2 обучений/4 состояний полностью равны прежним. Новых обучений не было; математика и defaults сохранены.
<!-- SECTION:FINAL_SUMMARY:END -->
