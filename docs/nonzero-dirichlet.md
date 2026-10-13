# Ненулевые Дирихле: математическая конструкция

TASK-23, 2026-10-13. Box 1D–3D, все грани Дирихле, совместимые IC/BC;
constant values или differentiable callbacks. Variable parameters и пользовательский
output transform пока исключены. Система не распознаёт heat по имени функции.

## Что такое lifting

Lifting — вспомогательная функция, которая продолжает заданные граничные
температуры внутрь области. Обозначим её $A(\mathbf{x},t)$. От неё требуется
совпасть с температурой стенки $h$ на границе $\Gamma$:

$$
A(\mathbf{x},t)\big|_{\Gamma}=h(\mathbf{x},t).
$$

Внутри области она не обязана решать уравнение теплопроводности. Это отличие
от искомого решения $u$: lifting обеспечивает BC, а сеть обучается по PDE.

Например, в 1D на отрезке $[a,b]$ задано две температуры $h_-(t)$ и $h_+(t)$.
Подходит обычная линейная интерполяция:

$$
\xi=\frac{x-a}{b-a},\qquad
A(x,t)=(1-\xi)h_-(t)+\xi h_+(t).
$$

В 2D–3D складываются интерполяции от всех пар противоположных граней, затем
вычитается повторный вклад пересечений. Пусть $P_i$ — интерполяция между гранями
оси $i$. При непрерывно совместимых traces:

$$
\begin{aligned}
A_{1D}&=P_1h,\\
A_{2D}&=P_1h+P_2h-P_1P_2h,\\
A_{3D}&=P_1h+P_2h+P_3h
-P_1P_2h-P_1P_3h-P_2P_3h+P_1P_2P_3h.
\end{aligned}
$$

При ограничении на грань $i$ члены без $P_i$ сокращаются с соответствующими
членами с $P_i$; остаётся $h_i$. Реализация заранее компилирует subsets осей
и sides: $3^d-1$ слагаемых на поле (2/8/26). Ось каждой проекции фиксируется
`full_like` на физической границе; tangential/time зависимости сохраняют AD.
Противоречивые значения на углах не усредняются: такие данные отклоняются
при обнаружении на проверочной сетке.

## Как lifting соединяется с начальным условием

Обозначения: $g$ — начальная температура, $N_\theta$ — выход сети,
$L_i=b_i-a_i$, $T=t_1-t_0$:

$$
\xi_i=\frac{x_i-a_i}{L_i},\qquad
\tau=\frac{t-t_0}{T},\qquad
B(\mathbf{x})=\prod_{i=1}^{d}4\xi_i(1-\xi_i).
$$

Вспомогательный background и физическое поле:

$$
\begin{aligned}
G(\mathbf{x},t)&=g(\mathbf{x})+A(\mathbf{x},t)-A(\mathbf{x},t_0),\\
u_\theta(\mathbf{x},t)&=G(\mathbf{x},t)+\tau B(\mathbf{x})N_\theta(\mathbf{x},t).
\end{aligned}
$$

При $t=t_0$ разность lifting и поправка сети равны нулю: $u_\theta=g$.
На любой грани $B=0$. Если $g|_\Gamma=h(t_0)=A(t_0)|_\Gamma$, то
$u_\theta|_\Gamma=h(t)$. Это условная математическая гарантия для совместимых
функций; проверка произвольных callback на сетке не доказывает тождество между
узлами. FP32 оставляет машинный остаток.

## Производные и обучение

Нижние индексы обозначают частные производные по физическим координатам:
$G_i=\partial G/\partial x_i$, $G_{ii}=\partial^2G/\partial x_i^2$.
Начальное время $t_0$ фиксировано:

$$
\begin{aligned}
G_t&=A_t(\mathbf{x},t),&G_{tt}&=A_{tt}(\mathbf{x},t),\\
G_i&=g_i+A_i(\mathbf{x},t)-A_i(\mathbf{x},t_0),\\
G_{ii}&=g_{ii}+A_{ii}(\mathbf{x},t)-A_{ii}(\mathbf{x},t_0).
\end{aligned}
$$

Для $u_\theta=G+\tau BN_\theta$ правило произведения даёт:

$$
\begin{aligned}
u_t&=G_t+\frac{B}{T}N+\tau BN_t,\\
u_{tt}&=G_{tt}+\frac{2B}{T}N_t+\tau BN_{tt},\\
u_i&=G_i+\tau(B_iN+BN_i),\\
u_{ii}&=G_{ii}+\tau(B_{ii}N+2B_iN_i+BN_{ii}).
\end{aligned}
$$

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

На единичном прямоугольнике, $\kappa=0.1$, точное решение состоит из двух частей:

$$
\begin{aligned}
b(\mathbf{x},t)&=1+\sum_{i=1}^{d}x_i^2+2\kappa d\,t,\\
v(\mathbf{x},t)&=e^{-\kappa d\pi^2t}\prod_{i=1}^{d}\sin(\pi x_i),\\
u_*(\mathbf{x},t)&=b(\mathbf{x},t)+v(\mathbf{x},t).
\end{aligned}
$$

$b$ — известный для этого benchmark полиномиальный фон. Он удовлетворяет heat,
поскольку $b_t=2\kappa d=\kappa\Delta b$. $v$ — затухающая пространственная мода:
её форма остаётся произведением синусов, а амплитуда экспоненциально уменьшается.
На стенках синусоидальная мода нулевая, поэтому $h=b|_\Gamma$. В `functions.py`
задаются только IC/BC/PDE без forcing; независимое аналитическое решение
используется после обучения, не передаётся optimizer.

## Что означает «ошибка затухающей моды»

В текущем коде это **ошибка полного решения, нормированная на размер моды**.
Сеть не обязана выдавать $v$ отдельным полем. Обе метрики используют один числитель:

$$
\begin{aligned}
\varepsilon_{\mathrm{full}}&=\frac{\lVert u_\theta-u_*\rVert_2}{\lVert u_*\rVert_2},\\
\varepsilon_{\mathrm{mode}}&=\frac{\lVert u_\theta-u_*\rVert_2}{\lVert v\rVert_2}.
\end{aligned}
$$

Норма считается по всем узлам общей пространственно-временной сетки, а не
только по последнему слою. Ошибка фона тоже входит в числитель; метрика не
отделяет её от ошибки моды и не измеряет только коэффициент синуса.
Эквивалентная интерпретация: сравнить $u_\theta-b$ с $v$, используя точно известный
фон benchmark. За пределами такого benchmark известного $b$ может не быть.

Если на сетке норма всего решения равна 100, норма моды — 1, а норма ошибки —
0.1, получим 0.1% общего L2 и 10% ошибки относительно моды. Большой фон скрывает
неточность динамической части в первой метрике. По мере затухания моды её
относительная ошибка может расти даже при постоянной абсолютной ошибке, поэтому
нужны также absolute RMSE/max error и разбивка по времени.

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

## Идеи удешевления: порядок проверки

Ниже предложения, не реализованные ускорения. До сравнений timings отделяем
background/lifting, его derivatives, сеть и optimizer. Число слагаемых само по
себе не доказывает долю GPU-времени или достижимый speedup.

1. **Объединить два прохода по одной геометрии.** Вес каждого слагаемого lifting
   зависит от пространства, но не от времени. Вместо двух отдельных сумм:

   $$
   A(\mathbf{x},t)-A(\mathbf{x},t_0)
   =\sum_{j}w_j(\mathbf{x})
   \bigl[h_j(\Pi_j\mathbf{x},t)-h_j(\Pi_j\mathbf{x},t_0)\bigr].
   $$

   Здесь знак inclusion-exclusion включён в $w_j$, $\Pi_j$ фиксирует координаты
   на соответствующем face/edge/corner. Projected columns и weights готовятся
   один раз на forward и используются для обоих времён. Два вызова произвольного
   callback всё ещё нужны: это не сокращение их числа вдвое. Все зависимости
   и derivatives сохраняются; иной порядок FP32 суммирования требует сравнения.

2. **Убрать повторные projections и произведения весов внутри одного forward.**
   Один projected column/нормированный фактор может обслуживать несколько
   слагаемых. Для 3D вместо повторных full_like и multiplication строится общий
   небольшой граф. Граф живёт один forward, без переноса между optimizer steps.
   Можно батчировать вызовы одного callback, но только при явном pointwise
   контракте: функция с зависимостью между строками batch при этом изменится.

3. **Кеш на фиксированной predict grid.** Для чистых неизменных IC/BC подготовить
   $G$ и $\tau B$ один раз; при следующих предсказаниях вычислять только сеть и
   восстановление $u$. Это полезно для повторного predict; cold подготовка всё
   равно оплачивается. Инвалидация при изменении grid/time/recipe/callback
   inputs/dtype/device. Сравнить один cold запуск и несколько warm проходов.

4. **Prepared derivatives для fixed objective обучения.** При IC/BC, не зависящих
   от обучаемых величин, на фиксированных PDE-точках подготовить $G_t$, $G_{ii}$,
   $B$, $B_i$, $B_{ii}$ (и значения для prediction). Для heat residual нужны
   первые time и вторые space derivatives. Если их действительно подставлять
   явно, геометрию и фон не придётся дифференцировать в каждой L-BFGS closure.
   Простой detach поля с последующим обычным autograd недопустим: исчезнут
   пространственные производные. Arbitrary PDE/higher/mixed derivatives требуют
   полного поддержанного набора либо eager fallback. Callback с обучаемыми
   Tensor из текущих тестов запрещает такой постоянный кеш без восстановления
   parameter gradients. Resampling инвалидирует подготовку.

5. **Явное простое продолжение вместо общего lifting.** В нашем benchmark можно
   задать $A=b$. Тогда

   $$
   A(\mathbf{x},t)-A(\mathbf{x},t_0)=2\kappa d(t-t_0),
   \qquad G=g+2\kappa d(t-t_0).
   $$

   Это та же математическая конструкция для данного benchmark, с короткой
   формулой вместо 26 слагаемых. В общем случае разные admissible lifting могут
   изменить interior background и класс поправок конечной сети: одинаковые BC
   не означают идентичную модель/траекторию обучения. Нужен optional контракт
   «smooth boundary extension», проверка traces и отдельная numerical comparison.
   Продолжение не обязано знать полное решение PDE. Нельзя автоматически считать
   face callback определённым и гладким внутри области.

Сначала пункты1/2: они сохраняют текущий recipe и не требуют нового пользовательского
API. Затем predict cache и ограниченный prepared heat objective. Explicit extension
даёт наиболее заметное сокращение в этом конкретном benchmark, но его общая
поддержка требует отдельного контракта. Compile/fusion — последующий эксперимент,
после локального профиля и проверки поддержки текущего стека.

Каждый кандидат сначала сравнивается на одинаковых weights/points: values,
first/second derivatives, independent physical PDE residual, parameter gradients,
IC/faces/edges/corners в float64/FP32. Затем короткий CUDA timing и парное
обучение. Проценты ускорения до эксперимента неизвестны; отсутствие пользы —
допустимый результат. Смягчать точность ради выигрыша не предлагается.
