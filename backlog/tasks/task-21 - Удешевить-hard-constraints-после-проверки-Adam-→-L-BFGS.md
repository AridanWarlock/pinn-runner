---
id: TASK-21
title: Удешевить hard constraints после проверки Adam → L-BFGS
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-13 13:28'
updated_date: '2026-10-13 17:04'
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
    https://github.com/AridanWarlock/pinn-runner/blob/codex/model-accuracy/docs/accuracy-experiments-2026-10-13.md
priority: high
type: spike
ordinal: 21000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Hard constraints улучшили heat3d, но на пяти seed увеличили медианное время обучения 6000 шагов с 56.32 до 76.58 s (+36%), тёплый инференс 152303 значений — с 19.43 до 46.87 ms (×2.41). Измерения: RTX 4070 Ti, Torch 2.1.0+cu118, Lightning 2.1.0. Цель — сохранить точные IC/BC и качество PDE-решения, удешевив представление u=g+τBN. Отдельно оценивать цену шага и время достижения заданной точности.

Очередность: после TASK-11 и сохранённой CUDA-проверки Adam → L-BFGS, включая сравнение при равном wall budget и closure/residual evaluations. Оптимизатор не менять в рамках этой задачи. Текущий eager hard path — контроль. Пользователь разрешил реализацию и GPU-эксперименты 2026-10-13; работа начата.

Вопросы исследования и математические ограничения:
• Аффинную нормализацию z=Ax+c можно точно свернуть в первый Linear: W_new=WA, b_new=Wc+b; постоянный масштаб выхода — в последний. Подготовленная predict-копия допустима. Изменение параметризации при train не гарантирует сохранение траектории Adam/L-BFGS и не допускает молчаливого переноса optimizer state.
• Координатно-зависимый τB нельзя универсально поглотить постоянными весами Linear. Hard layer сохраняет эту зависимость; перенос кода внутрь nn.Module сам по себе не доказывает ускорение.
• Fusion g/B/восстановления u может уменьшить kernel launches и промежуточные тензоры. Преобладание launch/memory overhead пока гипотеза; требуется профиль. Поддержка compiler/toolchain на текущем Windows-стеке проверяется, не предполагается.
• Кеш g/B для fixed-grid predict и известных производных для train требует инвалидизации при смене сетки/resampling, bounds, initial function, dtype/device и параметров задачи.
• Обезразмеривание heat даёт α=κT/L² и может улучшить обусловленность, но не гарантирует меньше FLOPs. Масштабы координатных производных и времени должны сохраниться.
• Явные производные hard transform могут позволить дифференцировать только N. Пример 1D: r=BN+sBN_s−α[g''+s(B''N+2B'N_ξ+BN_ξξ)], B=4ξ(1−ξ), B'=4−8ξ, B''=−8. Это пример эквивалентности, не универсальная замена произвольной PDE. Не делить residual на sB на IC/BC. Отсоединять кешированные g/B без восстановления их производных нельзя: это меняет PDE.
• Граничный базис меняет класс модели и оценивается отдельно от прозрачных ускорений.

Поставка: исследование и один небольшой обоснованный opt-in fast path, если польза подтверждена. Не требуется реализовать все идеи одновременно. Отрицательный воспроизводимый результат допустим. Пользовательский functions.py не должен требовать ручных g/B или переписывания PDE ради ускорения. Eager fallback и проверки применимости сохраняются. Обновление зависимостей не входит в scope: TASK-19 остаётся Deferred; TASK-15 исследует общий backend/fused optimizer, здесь фокус на hard transform и его производных.

Актуальный scope по запросу пользователя: теплопроводность и декартовы прямоугольные области 1D–3D. Численная матрица heat2d/heat3d; другие PDE, включая Burgers/advection, не входят в текущую приёмку. Геометрия/derivative execution не объявляются универсальным numerical default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 До работы зафиксированы CUDA-результаты TASK-11, контрольный checkpoint/config/revision, hardware, FP32/TF32 и seed. Профиль разделяет сеть, g/B, derivatives, optimizer/closure, launches, allocations и transfers; profiler отделён от итогового timing.
- [ ] #2 Для fusion, свёртки аффинных преобразований, кеша, обезразмеривания PDE и явных производных документированы применимость, эквивалентность либо изменение модели и результаты выбранных прототипов. Оценена верхняя граница ускорения по измеренной доле затрат; непроверенные возможности названы гипотезами.
- [ ] #3 Принимаемые fast paths совпадают с eager по значениям, первым/вторым координатным производным, PDE residual и parameter gradients в обоснованных float64/FP32 tolerances. Проверены IC, faces, edges/corners, interior/near-boundary и shifted bounds/time в 1D/2D/3D; деления на нулевой τB нет.
- [ ] #4 Кеш/подготовленная модель корректны при смене сетки/resampling, bounds, initial callback, dtype/device и recipe; checkpoint/predict/fine-tune contracts сохранены. Train fast path совместим с fixed objective внутри L-BFGS closure и честным учётом evaluations.
- [ ] #5 Windows RTX сравнение из C:\Projects\pinn-runner включает cold end-to-end с подготовкой/компиляцией/кешем, warm train step/closure и predict latency/throughput, peak allocated/reserved memory. Измерены окупаемость fixed-grid кеша, batch sizes 128/2048 и последний неполный batch.
- [ ] #6 Прозрачное ускорение сравнивается на одинаковом checkpoint/grid; траектории FP32 отдельно проверены попарно на seed 1234/4321/19/2026/73 для heat2d/heat3d в согласованном heat scope. Есть общий/final/worst-time L2, IC/BC, independent physical-autograd PDE holdout, counters и время до заранее заданной точности без analytic teacher. Другие PDE не считаются проверенными этой матрицей.
- [ ] #7 Принятие опирается на все seed, numerical gates, скорость и память, без ослабления порогов или смены FP32. Вариант выключаемый, eager fallback доступен; unsupported backend не выдаётся за compiled. Defaults не меняются до проверки комбинации.
- [ ] #8 Сохранён отчёт с исходными измерениями, ограничениями, отрицательными результатами и решением по вариантам; docs обновлены для поставленного fast path. Выполнены профильные тесты, необходимый full CPU check и Windows CUDA acceptance с точным названием набора. При отсутствии пользы допускается обоснованный отказ от внедрения.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Начать с профиля eager hard на текущем FP32 CUDA стеке и закрепить контроль TASK-11. 2. Проверить сокращение kernel launches в B через эквивалентный полином; доказать значения, производные и parameter gradients на shifted 1D–3D. 3. Принять только измеренно полезный opt-in backend с сохранением eager; далее подготовленный predict и ConstraintPlan по docs/hard-heat-plan.md. 4. Отдельно интегрировать расширенные heat BC; не объявлять эти этапы завершёнными в TASK-21.

Следующий блок: отдельный HardBoxDerivatives с точным правилом произведения и AD только для N/g; проверить float64/FP32 derivatives/residual/parameter gradients на shifted 1D–3D и границах. Сначала standalone CUDA comparison на одном checkpoint. Интегрировать в Equation только при подтверждённой пользе; никаких detach train cache или изменений функций пользователя.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Пользователь 2026-10-13 явно разрешил начать выполнение плана; прежнее ограничение description «не запускать» отменено новым поручением. Полный heat-план сохранён в docs/hard-heat-plan.md.

Первый блок: opt-in fused_predict (15c3133), warm heat3d checkpoint 3000 Adam+100 LBFGS, batch2048 46.72→43.25 ms (−7.43%), batch128 740.95→678.50 ms (−8.43%). Cold public runner 49a1650: median2.340→2.349 s, выигрыш отсутствует; все6 MAT совпали. Random4096 max field delta1.788e−7; training сохраняет прежний graph. Синтетический addcmul backward на3–4% медленнее, поэтому не используется при train. Полный CPU251, последующий быстрый232/19excluded, профильныйCPU12, Windows hard+hardware19passed. Полный profiler/heat derivative fast path ещё не завершены; task остаётся In Progress. Подробности docs/hard-execution-experiments-2026-10-13.md. Дальнейшая heat-интеграция отдельно TASK-22/23/24.

Интегрирован product_rule (9d1c98d): u_t=BN/T+tau BN_t; u_ii=g_ii+tau(B_ii N+2B_iN_i+BN_ii). Геометрия без деления на B, g/N остаются в AD; Equation callbacks пользователя не переписаны. 30 CUDA обучений:90k Adam calls,5000 LB calls,9576 closures,21,325,824 PDE visits. Fixed budget: train median2D33.46→29.62s (−11.47%),3D42.64→37.14s (−12.90%); отдельных seed точность хуже. Timing-only budget заранее зафиксирован a4f9be2: LB290/310; med3D finalL2 .317%→.110% при42.64→42.57s, global/final/PDE лучше10/10pairs. Full CPU262passed,mypy52; явный CUDA hardware8passed плюсderivatives6/9/10 и30train/predict/holdout. ПредыдущиеWindows19 безenv=cuda былиCPU, не CUDA acceptance. Отчёт docs/hard-derivative-experiments-2026-10-13.md. Task остаётся In Progress: CUDA kernel decomposition и true time-to-quality snapshots ещё не выполнены; defaults не изменены.
<!-- SECTION:NOTES:END -->
