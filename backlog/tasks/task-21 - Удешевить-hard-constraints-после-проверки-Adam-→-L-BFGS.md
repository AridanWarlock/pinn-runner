---
id: TASK-21
title: Удешевить hard constraints после проверки Adam → L-BFGS
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-13 13:28'
updated_date: '2026-10-17 08:34'
labels:
  - performance
  - hard-constraints
  - experiment
milestone: m-0
dependencies:
  - TASK-11
  - TASK-10
references:
  - pinnstorch/training/constraints.py
  - pinnstorch/models/net/neural_net.py
  - TASK-15
  - TASK-19
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-tasks/task-21.md
priority: high
type: spike
ordinal: 21000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Требования и критерии: docs/research-tasks/task-21.md. Сначала прочитать этот документ; подробную историю загружать только для конкретного вопроса.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 До работы зафиксированы CUDA-результаты TASK-11, контрольный checkpoint/config/revision, hardware, FP32/TF32 и seed. Профиль разделяет сеть, g/B, derivatives, optimizer/closure, launches, allocations и transfers; profiler отделён от итогового timing.
- [ ] #2 Для fusion, свёртки аффинных преобразований, кеша, обезразмеривания PDE и явных производных документированы применимость, эквивалентность либо изменение модели и результаты выбранных прототипов. Оценена верхняя граница ускорения по измеренной доле затрат; непроверенные возможности названы гипотезами.
- [ ] #3 Принимаемые fast paths совпадают с eager по значениям, первым/вторым координатным производным, PDE residual и parameter gradients в обоснованных float64/FP32 tolerances. Проверены IC, faces, edges/corners, interior/near-boundary и shifted bounds/time в 1D/2D/3D; деления на нулевой τB нет.
- [ ] #4 Кеш/подготовленная модель корректны при смене сетки/resampling, bounds, initial callback, dtype/device и recipe; checkpoint/predict/fine-tune contracts сохранены. Train fast path совместим с fixed objective внутри L-BFGS closure и честным учётом evaluations.
- [ ] #5 Windows RTX сравнение из C:\Projects\pinn-runner включает cold end-to-end с подготовкой/компиляцией/кешем, warm train step/closure и predict latency/throughput, peak allocated/reserved memory. Измерены окупаемость fixed-grid кеша, batch sizes 128/2048 и последний неполный batch.
- [ ] #6 Прозрачное ускорение сравнивается на одинаковом checkpoint/grid; траектории FP32 отдельно проверены попарно на seed 1234/4321/19/2026/73 для heat2d/heat3d в согласованном heat scope. Есть общий/final/worst-time L2, IC/BC, independent physical-autograd PDE holdout, counters и время до заранее заданной точности без analytic teacher. Другие PDE не считаются проверенными этой матрицей.
- [x] #7 Принятие опирается на все seed, numerical gates, скорость и память, без ослабления порогов или смены FP32. Вариант выключаемый, eager fallback доступен; unsupported backend не выдаётся за compiled. Defaults не меняются до проверки комбинации.
- [x] #8 Сохранён отчёт с исходными измерениями, ограничениями, отрицательными результатами и решением по вариантам; docs обновлены для поставленного fast path. Выполнены профильные тесты, необходимый full CPU check и Windows CUDA acceptance с точным названием набора. При отсутствии пользы допускается обоснованный отказ от внедрения.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Текущие границы и следующий вопрос — в docs/research-tasks/task-21.md. Перед реализацией уточнить план по актуальному коду.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
История сохранена в docs/research-history/task-21.md.
<!-- SECTION:NOTES:END -->
