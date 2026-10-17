---
id: TASK-30
title: Проверять и показывать план протокола до обращения к GPU
status: Done
assignee:
  - '@codex'
created_date: '2026-10-17 13:32'
updated_date: '2026-10-17 13:43'
labels: []
dependencies:
  - TASK-29
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-planning.md
priority: high
type: enhancement
ordinal: 30000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ошибки поддерживаемых случаев и сочетаний сейчас могут обнаруживаться только на Windows. Нужен план без Torch и сети; требования: docs/research-planning.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 plan без Torch, сети и записи файлов показывает матрицу, бюджет и границы оценки стоимости.
- [x] #2 Mac submit и Windows prepare используют одну предварительную проверку; неверные случаи, грани, архитектура и неподдерживаемые сочетания отклоняются до запуска.
- [x] #3 Лёгкий каталог и реестр моделей согласованы, конфигурации и математические эталоны сохранены.
- [x] #4 Профильные/общие проверки пройдены, существующий протокол и научные данные не изменены; передача обновлена.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Общий лёгкий каталог случаев и возможностей. 2. Проверка исполнимости и краткий план. 3. Подключение к submit/prepare/runner. 4. Проверки соответствия каталога и прежних протоколов без CUDA; документация.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Реализован общий offline plan до SSH и подготовки очереди. Каталог 19 случаев сохранён по независимым конфигурациям и 57 аналитическим значениям. Коммит f224016; полный набор 771 тест, профильный 89. Итог: docs/research-planning-result-2026-10-17.md. Нового GPU-обучения не было.
<!-- SECTION:FINAL_SUMMARY:END -->
