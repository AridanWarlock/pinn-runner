# Ненулевые Дирихле: математическая конструкция

TASK-23, 2026-10-13. Box 1D–3D, все грани Дирихле, совместимые IC/BC;
constant values или differentiable callbacks. Variable parameters и пользовательский
output transform пока исключены. Система не распознаёт heat по имени функции.

## Lifting

Пусть ξᵢ=(xᵢ−aᵢ)/Lᵢ, ℓᵢ⁻=1−ξᵢ, ℓᵢ⁺=ξᵢ. Pᵢ интерполирует между двумя
гранями оси i: умножает заданные face traces на ℓᵢ⁻/ℓᵢ⁺. На пересечении разных
граней traces обязаны совпадать; тогда композиции PᵢPⱼ граничных данных однозначны.
Boolean sum:

- 1D: A=P₁h.
- 2D: A=P₁h+P₂h−P₁P₂h.
- 3D: A=P₁h+P₂h+P₃h−P₁P₂h−P₁P₃h−P₂P₃h+P₁P₂P₃h.

При ограничении на грань i члены без Pᵢ сокращаются с соответствующими членами
с Pᵢ; остаётся hᵢ. Реализация заранее компилирует subsets осей и sides:
3^d−1 слагаемых на поле (2/8/26). Ось каждой проекции фиксируется full_like
на её физической границе; нормальная зависимость trace исключается, tangential
и time зависимости сохраняют AD. Никакого усреднения противоречивых углов нет.

B=∏4ξᵢ(1−ξᵢ), τ=(t−t₀)/T,
G=g(x)+A(x,t)−A(x,t₀), u=G+τBN.

При t₀ разность lifting нулевая, поэтому u=g. На любой грани B=0 и
совместимость g=h(t₀)=A(t₀) даёт u=h(t). Включение-исключение предотвращает
повторное сложение углов/рёбер; это не выбор количества BC samples.
Гарантия относится к непрерывно совместимым данным и математической арифметике;
FP32 округление оставляет машинный остаток.

## Производные и обучение

G_t=A_t(x,t), G_tt=A_tt(x,t).
G_i=g_i+A_i(t)−A_i(t₀), аналогично G_ii.

u_t=G_t+B N/T+τB N_t,
u_tt=G_tt+2B N_t/T+τB N_tt,
u_i=G_i+τ(B_iN+BN_i),
u_ii=G_ii+τ(B_iiN+2B_iN_i+BN_ii).

Product-rule adapter получает G derivatives через AD, а не использует старое
предположение G_t=0. Геометрия, lifting и IC не detach: производные loss по θ
сохраняют зависимости callback, если они содержат обучаемые Tensor. Начальное
время фиксировано full_like и не дифференцируется как текущее t. Полный physical
AD остаётся независимым holdout и fallback для более высоких производных.
Нет деления на B или τ; углы/initial не порождают сингулярность.

Нулевые constant traces используют прежний recipe без lifting: дополнительных
26 слагаемых в старом heat hot path нет. Новые nonzero callbacks требуют больше
вычислений; их ускорение здесь не объявляется. Runtime finite checks на GPU
в lifting не добавлены, чтобы не синхронизировать каждое слагаемое; конечность и
согласованность проверяются до обучения, затем обычные checks objective.

## Совместимость и checkpoint

Проверяется g=h при t₀ на всех face grids, попарные traces разных осей на
пересечениях во все config time nodes, конечность значений. Triple corners
входят в pairwise edge checks. Повторное условие одной грани отклоняется.
Это sampled validation: произвольную функцию между узлами не доказать.
Совместимость производных для C²,¹ heat на initial/boundary тоже не проверяется;
разрыв производных может замедлить обучение даже при совпадении значений.

Hard checkpoint identity для nonzero lifting включает source bytes и sorted
физические boundary specifications (field, axis, side, value/function). Вес loss
не входит в физический recipe. Для прежнего zero hard source-only identity
сохранена. Переключение execution adapter остаётся допустимым; изменение
boundary recipe — нет. Схема checkpoint не менялась.

## Независимый manufactured benchmark

На unit box, κ=0.1:

u*=1+Σxᵢ²+2κdt+exp(−κdπ²t)∏sin(πxᵢ).

Полиномиальная часть удовлетворяет u_t=2κd=κΔu; синусоидальная мода удовлетворяет
heat отдельно. На гранях мода нулевая, поэтому h=1+Σxᵢ²+2κdt. Начальное поле
включает полином и моду. В functions.py задаются только IC/BC/PDE без forcing;
независимая NumPy формула живёт в benchmark, не передаётся optimizer.
Relative L2 полного решения здесь зависит от большого background; поэтому
оценивать вместе с absolute max error, PDE RMSE и отдельно ошибкой относительно
нормы затухающей моды, а не делать вывод по одному проценту.

## CUDA результат и проверки

Revision `b46801c`, RTX4070Ti, Torch2.1.0+cu118, FP32/TF32 off. Оба case:
seed19, 3000 Adam+100 внешних L-BFGS шагов, product_rule. Аналитика только при
послеобучающей оценке; independent PDE holdout через полную AD физических полей,
2048 случайных точек. Quality gates прежние, не ослаблялись. `mode_relative_l2`
пока диагностическая метрика, отдельный автоматический порог на неё не вводился.

| Case | Общий L2, % | Mode L2, % | Final L2, % | Max error | PDE RMSE | IC RMSE | BC RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2D | 0.003327 | 0.02123 | 0.003607 | 0.0002675 | 0.001752 | 9.65e-8 | 2.44e-7 |
| 3D | 0.007369 | 0.09402 | 0.008553 | 0.0009315 | 0.004594 | 6.52e-8 | 4.05e-7 |

| Case | Train, s (cold runner) | Warm predict, ms | Points | Points/s (warm) | Cold predict, s | Peak allocated, MiB | LB closures |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2D | 52.33 | 83.93 | 52111 | 620870 | 2.450 | 23.74 | 192 |
| 3D | 133.62 | 795.60 | 152303 | 191433 | 3.206 | 27.39 | 202 |

Warm timing: batch2048, inputs already on GPU, 3 warmup+7 measured passes,
CUDA synchronize, median. Cold predict включает Python/runner/load/write MAT.
Training — отдельный public runner subprocess, включая запуск/подготовку/fit и
refinement; breakdown/counters/allocator reserved присутствуют в raw JSON.
Суммарно два обучения ~186 s, не полная30trial матрица. Один seed не даёт оценки
разброса или оснований для смены defaults. 1D проверена математическими тестами,
численные CUDA обучающие benchmark в этом блоке — только 2D/3D.

Общая face-based интерполяция дорогая: 3D background требует двух оценок lifting
по26 слагаемых на forward. Сравнивать её скорость как прозрачное ускорение zero
hard нельзя — это новый класс BC. Нулевые constant BC не вызывают lifting.
Конечные BC RMSE отражают FP32, а не ненулевой boundary penalty. Будущая
оптимизация может использовать известную smooth extension либо компиляцию
shared traces, но нельзя считать произвольный face callback определённым
внутри области и автоматически заменять им lifting.

Полный CPU: **283 passed, 118.89s**, Ruff/format и strict mypy58. После уточнения
диагностической benchmark метрики и identity-теста: быстрый **262 passed,
21 e2e deselected**; профильные mathematics/reference **32 passed**, отдельный
lifting **10 passed**. Первый full выявил2 ошибки старого теста: неизвестный
case молча использовал Burgers formula, wall callback получал interior вместо
face coordinates. Они исправлены; повторный full успешен.

Windows `PINN_TEST_DEVICE=cuda`: lifting+publicroundtrip **12 passed,27.63s**.
Шесть dim/dtype lifting случаев используют CUDA, четыре validation/identity
случая CPU; два publicroundtrip выполняют CUDA train/refine/predict/fine-tune
для zero и nonzero heat. Не называть все12 чистыми GPU-тестами.
Проверены shifted1D–3D, FP32/float64, faces/nearfaces/edges/corners, первые/вторые
производные всех осей и времени, loss gradients; несовместимые IC/traces и
повторные faces отвергаются. Checkpoint physical identity не меняется от
перестановки граней, меняется от другой совместимой wall function; смена executor
проходит predict/fine-tune.

Windows raw reports/checkpoints: `numerical-reports/dirichlet-001`, pytest log/xml
`numerical-reports/dirichlet-cuda.*`. Environment dirty вызван старыми untracked
pytest directories/installer; `git diff --exit-code HEAD -- run.py pinnstorch
benchmarks scripts` подтвердил неизменность исходников. SSH завершён.
[Полные сырые метрики](experiments/2026-10-13-nonzero-dirichlet.json).
