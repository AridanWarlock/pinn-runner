---
id: TASK-21
title: Удешевить hard constraints после проверки Adam → L-BFGS
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-13 13:28'
updated_date: '2026-10-18 06:07'
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
1. Проверить текущую Windows-среду без нагрузки: Kineto/CUDA activities, наличие CUPTI, Nsight и состояние существующей очереди; не трактовать файл блокировки как активную работу. 2. Зафиксировать наблюдения и ограниченный протокол профиля одинаковых весов, отделив синхронизированное время от событий ядер и накладных расходов профилировщика. 3. Проверить, допускает ли существующая очередь такой профиль; обход через отдельный SSH-запуск запрещён проектным skill. Не обновлять зависимости и не запускать длительное обучение.

4. Подключить ограниченный frozen-profile к той же очереди с отдельным строгим контрактом: два правила, одни проверенные веса, общий lock/таймаут/снимок. Без отдельного SSH-запуска нагрузки. 5. После CPU-проверок опубликовать протокол, выполнить одну диагностическую пару Nsight Systems без CPU sampling, сохранить успешную трассу либо точную причину отказа. Время под профилировщиком не выдавать за ускорение.
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

2026-10-17: analytic012 завершена и проверена: 20/20 обучений, 240 срезов, строгая точность 2D 5/5 у обоих исполнителей на двух случаях при L-BFGS 2000. Время analytic/eager: медиана 0,87193 (мода), 0,76684 (полином); память не выросла. На срезах до 600 мода не подтверждена ни одной ветвью; это дополнительный анализ после старта. Отчёт: docs/analytic-2d-confirmation-result-2026-10-17.md, результаты 3e34cba. Пользователь попросил остановиться в течение 10 минут перед сном: серия завершилась до срока, активных расчётов и автоматического возобновления нет. Продолжать только по новому поручению. Аудит оставшихся требований: docs/task-21-evidence-audit-2026-10-17.md; defaults прежние.

2026-10-18: пользователь возобновил исследование. Добавлена совместная матрица аналитического полного поля: смещённые неединичные области/время, 1D–3D, FP32/FP64, IC/грани/рёбра/углы/near-boundary, полный гессиан, PDE и градиенты весов; независимый точный контроль выходного смещения. Все 6 сочетаний прошли. Профильные численные: 75 passed; итоговый быстрый CPU: 751 passed, 50 e2e исключены; отдельно файловые e2e 6 passed; Ruff/mypy прошли. Производственный код и defaults прежние, обучение GPU не запускалось, критерии автоматически не отмечены. Отчёт: docs/analytic-geometry-result-2026-10-18.md. Следующий вопрос — ограниченный профиль CUDA и недостающие cold/warm/predict измерения.

2026-10-18: выполнена native013 — перспективная диагностическая пара Nsight Systems на одних весах analytic012, без обновлений. Torch сообщает LIBKINETO_NOCUPTI, доступны только CPU-активности. Nsight Systems установлен вне PATH, но оба запуска отказали до нагрузки: Microsoft-Windows-DxgKrnl provider requires administrator privileges. Оба exit 0 корректно не признаны успехом: нет workload/report/ядер, compare=false. Права, стек и defaults не менялись; повтор не запускался. Добавлен отдельный контракт 3 frozen-profile в ту же очередь (lock, Job Object, snapshot, timeout); полный CPU 818 passed за 301.09 с, профильные 29 passed, Ruff/strict mypy прошли. Код 73e00ef, протокол a3ac174. Отчёт: docs/native-profile-result-2026-10-18.md. Активных запусков нет. Следующий независимый вопрос — cold/warm/predict без внешнего профилировщика, через зарегистрированный режим очереди. Общая приёмка TASK-21 открыта.
<!-- SECTION:NOTES:END -->
