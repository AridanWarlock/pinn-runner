# ConstraintPlan: разделение рецепта и исполнения

TASK-22, 2026-10-13. Factory больше не содержит алгоритм проверки hard и его
сборку: это ответственность `runner/constraint_plan.py`. План указывает IC,
грани и оставшиеся loss, а также математический рецепт и его callable executor.
Объекты frozen dataclass защищают состав плана от случайной перезаписи; это не
глубокая иммутабельность всех Tensor/замыканий.

Математика прежняя:

$$
B(\mathbf{x})=\prod_{i=1}^d4\xi_i(1-\xi_i),\qquad
\tau=\frac{t-t_0}{T},\qquad u=g+\tau BN.
$$

При $t=t_0$ член
коррекции исчезает; на грани B=0, поэтому u=g. Нулевое Дирихле обеспечивается
при условии g=0 на всех гранях. Проверка g остаётся численной на сетке,
тождество callback между узлами не доказано. Формулы производных и порядок
арифметики eager/fused не изменены. Product rule сохраняет тот же AD graph для
производных более высокого порядка. Geometry/IC не отсоединены от AD.

Recipe `HardBoxInitial` больше не содержит флагов product_rule/fused_predict.
Исполнение представлено `ProductRuleHard(recipe)` либо `FusedPredictHard(recipe)`;
eager использует callable recipe непосредственно. Условие grad-enabled в
адаптерах различает train и predict, а не выбирает новую математическую модель.
Строка конфигурации разбирается один раз до обучения. Дополнительный callable
уровень — Python overhead без дополнительных Tensor operations/GPU transfers;
профильные проверки нужны, но длительная численная матрица здесь избыточна.

Checkpoint по-прежнему хранит математический kind и source signature; executor
не создаёт новых параметров state_dict. Старый checkpoint собирается через
новый factory и загружается без изменения схемы. Fine-tune остаётся обучением
из model weights, а не продолжением идентичной optimizer trajectory.

Config/functions.py и defaults не менялись. Новый план не объявляет произвольную
PDE heat по имени и не реализует новые BC. Это основа TASK-23/24, где потребуется
отдельный математический рецепт и проверки применимости.

## Проверки

Code revision `d251c49`. Полный `python scripts/check.py --full`: Ruff lint/format,
strict mypy 54 файла, **268 passed за 101.38 s**. Профильный CPU набор
`test_constraint_plan.py`, `test_hard_constraints.py`, `test_hard_derivatives.py`:
**26 passed**. Дополнительно explicit strict mypy двух обновлённых benchmark
скриптов прошёл. CPU e2e проверяет train/refine/checkpoint/predict/fine-tune;
fine-tune product-rule checkpoint через eager создаёт конечные веса.

Windows: `PINN_TEST_DEVICE=cuda`, pytest `test_hard_derivatives.py` и
`test_hard_product_rule.py`: **11 passed за 14.10 s**. Шесть dim/dtype derivative
случаев используют CUDA; четыре dispatch/fresh-graph случая остаются CPU;
public train/refine/predict/fine-tune выполняется CUDA. Поэтому эти 11 нельзя
называть одиннадцатью GPU-тестами. Логи/проверки:
`numerical-reports/constraint-plan-cuda.log`, `constraint-plan-cuda.xml` на
Windows. SSH завершён. Полная долгосрочная CUDA benchmark-матрица не повторялась.

Unsupported hard, отсутствие грани и несовместимое initial проверены прежними
тестами; новые plan-тесты подтверждают одинаковые гарантии для всех трёх
executors и соответствие plan loss реальному datamodule. Eager/product rule
проверены на сохранении значений, derivatives, residual и parameter gradients,
включая shifted bounds, faces/corners и fresh graph. Задача TASK-22 завершена;
новые математические BC остаются в TASK-23/24.
