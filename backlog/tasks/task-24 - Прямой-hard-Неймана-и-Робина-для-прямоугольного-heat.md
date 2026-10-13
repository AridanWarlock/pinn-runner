---
id: TASK-24
title: Прямой hard Неймана и Робина для прямоугольного heat
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
type: spike
ordinal: 24000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Граничные производные и физические потоки требуют иной hard-конструкции, сохраняющей свободу значения температуры. Смешанные грани и углы могут быть несовместимыми или негладкими; нельзя обещать общий tensor-product рецепт без доказательства.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Поддержанные классы Neumann/Robin имеют вывод точности исходных BC для u без лишнего Dirichlet; неподдержанные сочетания явно отклоняются
- [ ] #2 Знаки внешних нормалей и различие derivative/physical flux подтверждены; согласованность IC и пересечений проверяется
- [ ] #3 Этапы одной грани, 1D пар и поддержанных 2D–3D комбинаций подтверждены производными, manufactured CUDA решениями и balance metrics
<!-- AC:END -->
