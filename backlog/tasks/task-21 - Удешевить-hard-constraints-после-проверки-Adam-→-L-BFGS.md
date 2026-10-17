---
id: TASK-21
title: Удешевить hard constraints после проверки Adam → L-BFGS
status: Waiting CUDA
assignee:
  - '@codex'
created_date: '2026-10-13 13:28'
updated_date: '2026-10-17 20:06'
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
1. Закрыть пробел чистого перспективного сравнения исполнителей в 2D: затухающая мода и полиномиальный контроль, пять seed, eager/neumann_analytic. 2. Зарегистрировать одинаковые начальные веса/LHS 1024, Adam 2000, L-BFGS 2000, tolerance_change=1e-12; строгие пороги 008 не менять. Более длинный бюджет выбран до обучения по результатам 008/010. 3. Использовать неизменяемый уже проверенный код 1a4c22f, без изменений численного метода; отправить одну серию из 20 обучений через очередь. 4. Собрать все результаты, отдельно оценить абсолютную точность, парное качество, время до точности, память и скорость; не продлевать бюджет. 5. Сохранить отчёт и компактные CSV, обновить контекст и наблюдения по всплескам токенов. Defaults прежние.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
История сохранена в docs/research-history/task-21.md.

2026-10-17: запущена density010, 5 пар 1024/4096 вложенных равномерных PDE-точек, по 60 с обучения. Код fdbc37c627d140d71d9ae8094412f52dfa70a57a; протокол 2fb0d0ad0d65c735e7e05e7e24d2710b9f0cb5ef:docs/experiments/2026-10-17-density-protocol.json. Математика и критерии: docs/density-neumann-protocol-2026-10-17.md. Полный CPU: 791 passed; заключительный быстрый: 744 passed, 50 e2e исключены; дополнительные профильные проверки и короткая CPU-проверка независимого оценщика прошли. Подтверждён running; численного результата ещё нет.

2026-10-17: density010 завершена, проверены 10/10 обучений и 100 срезов из чистого снимка fdbc37c; обе плотности 1024/4096 подтвердили строгую точность 5/5 за 60 с. Для 4096 медианное парное отношение PDE RMS 0,724, памяти 1,785; наблюдаемое подтверждение раньше в 4/5 пар, разрешение 5 с. Необходимость плотной выборки не доказана, defaults прежние, общая приёмка TASK-21 остаётся открытой. Полный отчёт и ограничения: https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/density-neumann-result-2026-10-17.md (результаты c12dfc4). Фоновых запусков серии нет.

2026-10-17: подготовлена, но НЕ запущена серия analytic011: два контроля 3D, пять seed, eager/neumann_analytic, 20 обучений; бюджет и пороги 009. Код 1a4c22f, протокол 62bb738; обоснование docs/analytic-3d-replication-protocol-2026-10-17.md. CPU: 73 профильных passed; быстрый 745 passed, 50 e2e исключены; Ruff/mypy прошли. Автоматическая проверка отклонила git push в origin из-за неподтверждённого разрешения на передачу изменений в GitHub. Коммиты пока только локально; submit не выполнялся. Требуется явное разрешение пользователя на публикацию этих коммитов в AridanWarlock/pinn-runner, codex/research-tools.

2026-10-17: пользователь явно разрешил push в AridanWarlock/pinn-runner; код 1a4c22f и протокол 62bb738 опубликованы. Очередь приняла analytic011, PID запуска 9244: 20 обучений, два контроля 3D, пять seed, eager/neumann_analytic. Ожидаются полный пакет и независимая оценка; численного вывода пока нет.

2026-10-17: analytic011 завершена, собраны и проверены 20/20 обучений и 180 срезов на чистом снимке 1a4c22f. На двух контролях 3D обе ветви подтвердили прежние пороги 009 на 5/5 seed. Медианные отношения времени analytic/eager: 0,85295 (мода), 0,86395 (полином); память около 0,9923, все парные критерии пройдены. Defaults прежние; следующая граница — чистое сравнение исполнителей 2D, общая приёмка TASK-21 открыта. Отчёт: https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/analytic-3d-replication-result-2026-10-17.md (a759cdb). Наблюдения по расходу контекста: docs/research-context-efficiency-2026-10-17.md. Фоновых обучений серии нет.

2026-10-17: запущена analytic012, протокол e27f96e, неизменяемый код 1a4c22f. Два контроля 2D × пять seed × eager/analytic = 20 обучений; Adam 2000, L-BFGS 2000, LHS 1024, строгие пороги 008. Больший бюджет выбран до запуска, по 008/010; изменения только документальные, повторный CPU-набор не запускался, plan прошёл. Результаты ожидаются; defaults прежние. Наблюдение и сбор повторно используют локальные сценарии 011 с заменой идентификатора, вместо чтения/генерации полного кода заново.
<!-- SECTION:NOTES:END -->
