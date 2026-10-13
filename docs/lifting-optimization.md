# Удешевление Dirichlet lifting: общая геометрия

TASK-25, 2026-10-13. Математический рецепт, PDE, optimizer и checkpoint identity
не изменяются. Исходный двухпроходный `DirichletLifting.__call__` оставлен как
численный контроль и обычный callable. Compiled lifting использует `difference`.
Пользовательские callable lifting сохраняют прежний fallback.

## Равенство, на котором основано сокращение

В каждом слагаемом lifting пространственный вес не зависит от времени. Поэтому
разность двух сумм можно записать как сумму разностей:

$$
\begin{aligned}
A(\mathbf{x},t)&=\sum_j w_j(\mathbf{x})h_j(\Pi_j\mathbf{x},t),\\
A(\mathbf{x},t)-A(\mathbf{x},t_0)
&=\sum_jw_j(\mathbf{x})
\left[h_j(\Pi_j\mathbf{x},t)-h_j(\Pi_j\mathbf{x},t_0)\right].
\end{aligned}
$$

Знак включения-исключения входит в вес. Проекция фиксирует выбранные координаты
на гранях, рёбрах или углах. Для математической проверки это точное равенство;
FP32 меняет порядок суммирования и может изменить округления и траекторию обучения.

В текущем forward общие projected columns и weights живут в prefix cache.
Если уже создана проекция на одну грань, пересечение с другой гранью строится из
неё заменой одной координаты и умножением на ещё один фактор. Столбцы границ и
нормированные нижние/верхние множители создаются один раз. Термы компилируются
по возрастанию числа проекций, поэтому родительский prefix уже существует.
Одна геометрия может обслуживать несколько физических полей.

В 3D на поле остаются26 разностей trace: 52 вызова callback (по два времени),
а не26 вызовов. Число fixed boundary columns уменьшается до6 на forward,
вместо суммарного многократного full_like в двух проходах. Уменьшаются также
повторные normalized factors и prefix multiplication; независимая PDE всё равно
требует вычисления необходимых coordinate derivatives.

## Сохранение autograd и контракт

Кеш не переносится между forward/optimizer steps и не содержит detach. Все
weights остаются функциями исходных Tensor coordinates. Callback получает ту же
проекцию и тот же размер batch; батчирование нескольких вызовов не применяется.
Parameter gradients в initial/trace сохраняются, включая искусственный trainable
Tensor из математического теста. IC/BC callbacks должны быть детерминированными,
без in-place изменения входов и зависимости от порядка вызовов: перемежение
вычислений двух времён отличается от исходного двухпроходного порядка.

В начальный момент оба callback вызываются на одинаковом времени и разность
нулевая. На гранях работает прежнее правило inclusion-exclusion при непрерывной
совместимости traces и initial. Sampled compatibility остаётся sampled; ускорение
не превращает её в доказательство произвольной функции между узлами.
Zero constant Dirichlet не вызывает lifting. Математический класс сети и source/BC
checkpoint signature не меняются; сравнивается тот же checkpoint без переобучения.

## Протокол проверки

CPU reference: двухпроходный callable wrapping bypasses optimized difference;
сравниваются values, первые/вторые derivatives, parameter gradients, IC/faces/
nearfaces/edges/corners, shifted1D–3D и FP32/float64. Сравнение product_rule с
полным physical AD теперь также использует старый background как контроль.

CUDA benchmark `scripts/benchmark_lifting.py`: прежние checkpoints TASK-23,
одинаковые weights и physical points (seed20261013), batch128/2048/17. Измеряет
product-rule residual+backward и no-grad forward. Три warmup, пять samples по10
вызовов; порядок baseline/shared обращается во втором повторе. GPU synchronize
перед/после measured sample; profiler не включён. Сохраняются max differences,
gradients, peak allocated, checkpoint SHA и environment.

Это warm batch timing без optimizer update, не время достижения точности и не
cold end-to-end predict. Выигрыш одной операции нельзя автоматически объявлять
выигрышем полного обучения. Публичные train/refine/predict/fine-tune roundtrip
проверяются отдельно короткими e2e. Полная30trial матрица здесь не повторяется.

## Измерения RTX4070Ti

Revision `50db44c`, Torch2.1.0+cu118, FP32/TF32 off. Медиана двух повторов
(в каждом — медиана пяти samples); числа ниже — миллисекунды на batch.
Train здесь означает только residual+backward, без optimizer update.

| Dimension | Batch | Train baseline → shared, ms | Снижение времени | Predict baseline → shared, ms | Снижение времени |
|---|---:|---:|---:|---:|---:|
| 2D | 128 | 13.587 → 11.606 | 14.58% | 3.315 → 2.353 | 29.02% |
| 2D | 2048 | 13.628 → 11.700 | 14.15% | 3.297 → 2.334 | 29.21% |
| 2D | 17 | 13.869 → 11.828 | 14.72% | 3.293 → 2.362 | 28.28% |
| 3D | 128 | 38.537 → 29.641 | 23.09% | 10.848 → 6.958 | 35.86% |
| 3D | 2048 | 39.231 → 30.196 | 23.03% | 10.721 → 6.848 | 36.13% |
| 3D | 17 | 38.756 → 29.902 | 22.85% | 10.620 → 6.764 | 36.30% |

Max FP32 field difference — 2.861e-6; residual difference — 2.384e-7;
parameter gradient difference — 6.194e-8. Все assert tolerances пройдены.
Градиенты не совпадают бит-в-бит; равные математические expressions округляются
по-разному. Это не эксперимент accuracy improvements или time-to-quality.

Checkpoint SHA256:

- 2D: `4dcd2c816243d32ab60d172669b071ce8fcdd02b8be43189d919ef3e50bb2d86`.
- 3D: `4d6340fe2e44695bb52cea39cb0f1e2eaf21ed74dbecc1f7bb3210e5bbfd15e6`.

Full CPU: **283 passed,113.89s**, Ruff/format и strict mypy58. Explicit strict mypy
нового benchmark —1filepassed; профильные lifting —10passed.
Windows `PINN_TEST_DEVICE=cuda`: **12passed,27.11s** (6CUDA math,4CPUvalidation,
2CUDA publicroundtrip zero/nonzero heat train/refine/predict/fine-tune).
Короткие timings использовали прежние weights без переобучения. Полная long
matrix не запускалась; численные defaults optimizer/constraints не менялись.
Math recipe shared difference автоматически применяется к compiled lifting.

Windows evidence: `numerical-reports/lifting-shared-cuda.log/xml`,
`lifting-shared-{2d,3d}.json`. [Полные timings](experiments/2026-10-13-lifting-shared.json).

## Public predict на полной сетке без нового обучения

`--measure-existing numerical-reports/dirichlet-001`: те же TASK-23 checkpoint,
новая code revision50db44c, полный public predict + physical AD holdout.
Оба case прошли прежние quality gates. Поля training_seconds/counters в этом
raw report унаследованы от старого обучения, это не новые training timings.

| Case | Warm baseline → shared, ms | Points | Cold baseline → shared, s | Mode L2 shared, % | PDE RMSE shared |
|---|---:|---:|---:|---:|---:|
| 2D | 83.93 → 60.23 | 52111 | 2.450 → 2.414 | 0.0212285 | 0.00175215 |
| 3D | 795.60 → 509.46 | 152303 | 3.206 → 2.887 | 0.0940245 | 0.00459436 |

Снижение warm времени полной сетки —28.24%/35.96%. Baseline whole-grid timing
взят из предыдущего TASK-23 report, а не из чередующегося matched experiment:
основная paired performance evidence — batch timing выше. Cold здесь по одному
запуску каждого варианта, без доказательства величины стабильного cold выигрыша.
BC RMSE shared —8.52e-8/6.77e-8; это изменение FP32 округлений, не обучение
более точной модели. IC unchanged. Values/PDE quality практически прежние.
[Public predict raw JSON](experiments/2026-10-13-lifting-predict.json).
SSH закрыт. Windows source diff относительно HEAD чист; environment dirty
вызван прежними untracked directories/installer. Память и времена подготовки
входят в соответствующие raw metrics; нового persistent cache/compile нет.

Вывод: принять shared geometry/difference для compiled lifting. Математический
recipe прежний, ускорение подтверждено на сохранённых 2D/3D weights и short public
roundtrip. Полное время обучения и time-to-quality после этой оптимизации не
измерены, поэтому23% train-operation gain не объявляется23% train-run gain.
Следующие persistent caches/explicit extensions остаются отдельными предложениями.
