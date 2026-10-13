---
id: TASK-23
title: Точные ненулевые Дирихле и IC для heat в прямоугольнике
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-13 15:57'
labels:
  - hard-constraints
  - heat
dependencies:
  - TASK-22
documentation:
  - docs/hard-heat-plan.md
priority: high
type: feature
ordinal: 23000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Нулевые Dirichlet грани не покрывают заданную температуру стенок, включая её изменение во времени. Требуются совместимые начальное поле и граничные данные без участия аналитического решения в обучении.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ненулевые временные BC и IC выполняются конструкцией в 1D–3D; несовместимые данные на пересечениях выявляются
- [ ] #2 Поле, первые и вторые производные и parameter gradients проверены на смещённых областях, гранях, рёбрах и углах
- [ ] #3 Manufactured heat решения подтверждены на CUDA; checkpoint и документация отражают поддержанные условия
<!-- AC:END -->
