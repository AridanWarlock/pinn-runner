---
id: TASK-24
title: Прямой hard Неймана и Робина для прямоугольного heat
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-13 15:57'
updated_date: '2026-10-13 19:43'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Проверить 1D проекцию Эрмита на kernel двух Robin operators (включая Neumann): QF=F-sum(phi_s L_s F), без фиксации температуры. 2. Проверить shifted bounds, normal signs, derivatives/parameter gradients и exact BC для manufactured fields. 3. Затем отдельная интеграция IC/time lift/ConstraintPlan и CUDA manufactured solve; многомерные пары требуют operator trace compatibility, не равенства flux values.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Первый этап 9ea7b50: IntervalRobinProjection с кубическим базисом Эрмита, однородные операторы Неймана/Робина на двух концах без фиксации температуры. Проверены знаки нормалей, сдвинутый отрезок, FP32/float64, производные1/2поx/t, градиент невязки теплопроводности, повторная проекция и предсказание с внешним отключением графа. БыстрыйCPU269прошли/21e2eисключён,Ruff,mypy59;профильныеCPU7прошли;Windows7прошли2.04s(6GPUчисленных+1CPUпараметры). Документdocs/robin-projection.md. Вconfigнеподключено;интеграцияIC,обучение,сохранение/загрузкаи2D/3Dещёпредстоят. Задачавработе,ACнеотмечены.
<!-- SECTION:NOTES:END -->
