---
id: TASK-39.9
title: 'Проверить суррогат для k(x,t)'
status: To Do
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 08:31'
labels: []
dependencies:
  - TASK-39.5
  - TASK-39.8
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-research-roadmap.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-terminology.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-validation.md
parent_task_id: TASK-39
priority: medium
ordinal: 48000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Объединить пространственную и временную изменчивость после раздельных проверок. Допускается начать с разделимого аналитического семейства, но его успех не означает освоения неразделимых функций.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Подготовлены аналитические пространственно-временные случаи и проверена причинная постановка; разделимые и неразделимые семейства явно различены.
- [ ] #2 До обучения определены класс гипотез, новые функции, бюджет и абсолютные цели; общая сеть оценена на новых гипотезах.
- [ ] #3 Проверены физические производные, потоки, точность, стоимость и файловая воспроизводимость; отчёт ограничивает вывод принятым классом.
<!-- AC:END -->
