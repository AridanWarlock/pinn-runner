# API файлового PINN-решателя, версия 1

Задача состоит из `config.yaml` и `functions.py`. Для `predict` и `retrain` нужен
checkpoint. `data.mat` не читается: обучение использует уравнения, обязательные
начальные условия и заданные границы. Эталонное решение можно отдельно использовать
для сравнения, но оно не входит в текущий API.

Поддерживаются непрерывное время, прямоугольная область 1D/2D/3D, несколько полей,
несколько уравнений и семейства задач со скалярными входными параметрами.
Восстановление неизвестных коэффициентов по наблюдениям не реализуется.

## Полный исполняемый пример

Это 2D теплопроводность с коэффициентом в диапазоне 0.1–0.5. Три шага предназначены
для проверки запуска; этого недостаточно для решения физической задачи.

<!-- executable-config -->
```yaml
version: 1
domain:
  space:
    x:
      interval: [0, 1]
      points: 8
    y:
      interval: [0, 1]
      points: 6
  time:
    interval: [0, 0.5]
    points: 3
  parameters:
    conductivity:
      interval: [0.1, 0.5]
      points: 3
network:
  hidden_layers: [16, 16]
training:
  epochs: 3
  steps: 3
  seed: 1234
  batch_size: 16
  optimizer: {learning_rate: 0.01}
  collocation: {sampling: grid}
validation: {sampling: grid}
boundaries:
- {kind: dirichlet, field: T, axis: x, side: both, value: 0}
- {kind: dirichlet, field: T, axis: y, side: both, value: 0}
runtime: {device: cpu, progress: false}
prediction:
  parameters:
    conductivity: [0.25]
```

<!-- executable-functions -->
```python
"""2D heat equation using the public problem language."""

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def heat(eq: Equation) -> torch.Tensor:
    return eq.dt("T") - eq.parameters["conductivity"] * eq.laplacian("T")


def initial(c: Coordinates) -> TensorFields:
    return {"T": torch.sin(torch.pi * c.x) * torch.sin(torch.pi * c.y)}


problem = Problem(fields=("T",), equations={"heat": heat}, initial=initial)
```

```sh
python run.py --mode train --input-dir ./task --output-dir /tmp/pinn-result
python run.py --mode predict --input-dir ./task --output-dir /tmp/pinn-prediction --checkpoint /tmp/pinn-result/checkpoints/final.ckpt
python run.py --mode retrain --input-dir ./task --output-dir /tmp/pinn-continued --checkpoint /tmp/pinn-result/checkpoints/final.ckpt
```

Относительный путь checkpoint считается относительно **input-dir**. Команды выше
используют абсолютный путь. По умолчанию директории `/task_data`
и `/task_output`, имя checkpoint внутри input-dir — `checkpoint.ckpt`.
`retrain` загружает веса и запускает новый оптимизатор; состояние оптимизатора и
счётчики прежнего обучения не восстанавливаются.

## Конфигурация

В YAML нет Python import paths, `_target_`, классов Lightning и списков внутренних
sampler-объектов. Неизвестные ключи вызывают ошибку с путём к настройке.
`version: 1` обязателен. Старый Hydra-формат больше не поддерживается.

### Область и параметры

`domain.space` содержит `x`, опционально `y`, затем `z`; `z` требует `y`.
`domain.time` задаёт время. Каждая ось имеет возрастающий конечный `interval`
из двух чисел и целое `points >= 2`. Начальные условия задаются при нижней
границе временного интервала. `domain.parameters` необязателен.

Имена параметров — ASCII-идентификаторы без начального `_`, отличные от
`x/y/z/t`, полей и уравнений. Порядок входа сети фиксирован:
`x[,y,z], параметры в порядке YAML, t`. Пространственные операторы не включают
параметры. В каждом batch параметр представлен столбцом Tensor `(N, 1)`;
разные строки могут иметь разные значения.

Полная сетка — декартово произведение пространства, параметров и времени.
Число точек быстро растёт в 3D и при нескольких параметрах. Для PDE можно
использовать LHS, но сетки границ и предсказания по-прежнему строятся полностью.

### Сеть

`network.hidden_layers: [32, 32]` — значение по умолчанию. Непустой список
положительных целых чисел задаёт скрытые слои с tanh. Размер входа берётся
из области, выходы — из `Problem`. Их не нужно вручную согласовывать с YAML.

### Обучение

| Настройка | По умолчанию | Смысл |
| --- | --- | --- |
| `training.epochs` | 100 | Положительное максимальное число эпох |
| `training.steps` | null | Дополнительное ограничение числа шагов оптимизатора |
| `training.seed` | 1234 | Неотрицательное целое; 0 допустим, null отключает установку seed |
| `training.batch_size` | null | Все точки каждого условия; иначе положительный размер batch |
| `training.initial_points` | null | Все начальные точки; иначе случайная подвыборка без повторений |
| `training.optimizer.name` | adam | adam или adamw |
| `training.optimizer.learning_rate` | 0.001 | Положительный learning rate |
| `training.optimizer.weight_decay` | 0 | Неотрицательный weight decay |
| `training.collocation.sampling` | grid | grid или lhs |
| `training.collocation.points` | отсутствует | Обязателен и положителен для lhs; запрещён для grid |

```yaml
training:
  epochs: 1000
  batch_size: 512
  collocation:
    sampling: lhs
    points: 10000
validation:
  sampling: lhs
  points: 2000
```

`validation` использует те же настройки выборки; по умолчанию grid.
`validation: false` или `null` отключает проверку. Без эталонных данных `val/loss`
измеряет невязки PDE, а не ошибку решения. Это не гарантия качества начальных
и граничных условий. Оптимизируется сумма MSE начальных условий, уравнений и
граничных ограничений. Короткие наборы условий повторяются, чтобы пройти самый
длинный набор; размер batch сам по себе не является физическим весом условия.

### Граничные условия

`boundaries` — список, по умолчанию пустой. Общие ключи:

| Ключ | Смысл |
| --- | --- |
| `kind` | dirichlet, neumann, robin или heat_flux |
| `field` | Имя физического поля из `Problem.fields` |
| `axis` | Существующая пространственная ось x/y/z |
| `side` | lower, upper или both; по умолчанию both |
| `value` | Конечное число, по умолчанию 0 |
| `value_function` | Вместо value: имя функции из `Problem.boundary_values` |
| `weight` | Положительный вес MSE, по умолчанию 1 |

Операторы при внешней нормали `n`:

| kind | Невязка | Дополнительные ключи |
| --- | --- | --- |
| dirichlet | u − g | — |
| neumann | ∂u/∂n − g | — |
| robin | αu + β∂u/∂n − g | alpha и beta, оба по умолчанию 1; не оба 0 |
| heat_flux | −k∂u/∂n − g | conductivity: положительное число либо имя положительного параметра; по умолчанию 1 |

На lower нормаль имеет знак −1, на upper +1. `alpha/beta` разрешены только
для Robin, `conductivity` — только для heat_flux.

Все границы Дирихле одного поля объединяются. Угол/ребро учитывается один раз,
значения в пересечении должны совпадать с допуском 1e-6, вес берётся максимальный.
Производные условия используют внутренность каждой грани, без углов/рёбер:
там нельзя выбрать единственную нормаль. Это относится и к Robin с beta=0.
В 1D торцевые точки допустимы. Если в 2D/3D после исключения пересечений точек
не осталось, runner просит увеличить tangential points. Пересечение грани
с начальным временным срезом не исключается: начальное и граничное условия —
разные физические ограничения. Их совместимость остаётся обязанностью автора задачи.

```yaml
boundaries:
  - kind: dirichlet
    field: T
    axis: x
    value_function: wall_temperature
  - kind: neumann
    field: T
    axis: y
    value: 0
```

Функция значения получает `Coordinates` со всей сеткой, возвращает число
либо конечный столбец `(N, 1)`. Значения вычисляются перед обучением и становятся
фиксированными целями, а не частью градиентного графа сети.

```python
def wall_temperature(c: Coordinates) -> torch.Tensor:
    return torch.ones_like(c.t) * 10

problem = Problem(
    fields=("T",), equations={"heat": heat}, initial=initial,
    boundary_values={"wall_temperature": wall_temperature},
)
```

### Предсказание и runtime

`prediction.parameters` задаёт непустой список различных конечных значений для
каждого нужного параметра. Неуказанные параметры используют сетку domain.
Этот раздел влияет только на `predict`. Значение 0.25 допустимо, даже если
обучающая сетка содержала 0.1, 0.3 и 0.5. Нормализация берётся из checkpoint.

Выход за область checkpoint по пространству, времени или параметрам запрещён.
`prediction.allow_extrapolation: true` явно разрешает его; это не гарантирует
точность вне обучающей области. Имена и порядок входов должны совпадать всегда.
Архитектура сети должна соответствовать весам; загрузка строгая.

| Настройка | По умолчанию | Смысл |
| --- | --- | --- |
| `runtime.device` | cpu | cpu, cuda, mps или auto |
| `runtime.devices` | 1 | Положительное число устройств или список неотрицательных индексов |
| `runtime.precision` | 32-true | 32-true, 16-mixed, bf16-mixed; поддержка зависит от устройства |
| `runtime.progress` | false | Индикатор прогресса |
| `runtime.metrics` | true | CSV-метрики Lightning |
| `runtime.save_training_predictions` | true | Дополнительное предсказание после train/retrain |

Для RTX: `device: cuda`, `devices: [0]`; начните с `precision: 32-true`.
Для Mac: CPU проверен тестами; MPS можно выбрать явно, но он не проверен этим
набором. Mixed precision выполняется средствами Lightning, отдельного самописного
AMP/JIT/CUDA Graph режима больше нет. Высокие производные PDE требуют проверки
численной устойчивости выбранной точности.

## functions.py: Problem и операторы

Файл экспортирует **один объект `problem = Problem(...)`**. Имена локальных функций
произвольны. `initial` обязателен: callback возвращает все физические поля.
Все вычисления пользовательских callbacks выполняются в Torch, без NumPy в PDE.
Callbacks должны работать поточечно и не менять поля/координаты inplace: это
сохраняет корректность autograd и кеша производных.
Скаляры начальных условий записывайте через `torch.full_like(c.t, value)`.

`Coordinates` содержит `.x/.y/.z` для существующих осей, `.t`, `.spatial` и
`.parameters` и `.coefficients`. Все значения — Tensor `(N, 1)`. `.variable(name)` возвращает
координату/параметр по имени. Отсутствующая ось не создаётся автоматически.

Каждое уравнение возвращает **невязку**, которая должна быть нулём: Tensor `(N, 1)`
на том же устройстве. Система не вычисляет правую часть автоматически.
Например, с источником `Q`: `e.dt("T") - k * e.laplacian("T") - Q`.
Можно задать несколько уравнений и полей через словарь и tuple.

| Оператор | Результат |
| --- | --- |
| `e.field("T")` | Значение поля |
| `e.dt("T")` | Первая производная по времени |
| `e.diff("T", "x", order=2)` | Производная заданного порядка по оси |
| `e.gradient("T")` | Словарь первых производных по физическим осям |
| `e.laplacian("T")` | Сумма вторых пространственных производных |
| `e.divergence({"x": fx, "y": fy})` | Дивергенция вектора со всеми физическими компонентами |
| `e.diff(tensor, "x")` | Производная произвольного выражения |

Смешанная производная: `e.diff(e.diff("T", "x"), "y")`. Чувствительность по
входному параметру: `e.diff("T", "conductivity")`. Эти производные не входят в
laplacian. Производные именованного поля кешируются внутри одного вызова PDE
для batch; между шагами кеш не сохраняется.

### Пространственно меняющийся коэффициент

Формула c(x,y) уже может возвращать Tensor. Для консервативной теплопроводности
нужна дивергенция потока, которая учитывает производные коэффициента:

```python
def heat(e: Equation) -> torch.Tensor:
    c = 1 + e.coordinates.x
    flux = {axis: c * derivative for axis, derivative in e.gradient("T").items()}
    return e.dt("T") - e.divergence(flux)
```

Это отличается от `c * laplacian(T)` при пространственно меняющемся c.
Теперь именованный коэффициент можно задать через `Problem.coefficients` либо
таблицу NPZ в разделе `coefficients` YAML. PDE читает `e.coefficients[name]`;
heat_flux принимает это имя в conductivity. Таблицы интерполируются в Torch
с производными по координатам. Полный формат, ограничения гладкости и правила
checkpoint описаны в [coefficients.md](coefficients.md).

Fixed table задаёт конкретную задачу. Одна сеть для произвольного нового поля
как независимого входа пока не поддерживается.

### Поля из потенциала

Если сеть должна возвращать потенциал, укажите `network_fields` и `transform`.
Transform получает Equation с сырыми выходами сети, возвращает все физические
поля. После преобразования PDE работает с физическими полями. Производные
разрешены и в predict, поэтому градиентный режим здесь сохраняется.

```python
def from_potential(e: Equation) -> TensorFields:
    return {"u": e.diff("psi", "y"), "v": -e.diff("psi", "x")}
```

Тогда `fields=("u", "v")`, `network_fields=("psi",)`,
`transform=from_potential`; initial возвращает u и v, equations задают их PDE.
Имена уравнений не должны совпадать с физическими/сетевыми полями.

## Результаты и ошибки

Train/retrain сохраняют `checkpoints/best.ckpt`, `last.ckpt` и `final.ckpt`.
Best выбирается по val/loss, либо train/loss без validation. Final содержит
последние веса до дополнительной проверки/предсказания. `predictions.pkl`, если
включён, использует best; это может быть другой шаг, чем final.

Predict сохраняет `predictions.mat` и `prediction.json`. В MAT координаты и
параметры — столбцы `(S, 1)`, время `(Nt, 1)`, поля `(S, Nt)`.
S — число комбинаций пространства и запрошенных параметров. Порядок — ij/C,
параметры после пространства, время меняется быстрее всего при flatten.
Поля MAT и тензоры pickle сохраняются в float32 на CPU, чтобы результаты можно
было читать и без CUDA. JSON содержит имена входов, размер поля, batch_size, обучающие и запрошенные
границы. Помимо физических полей transform может сохранять сетевые потенциалы.

Все режимы сохраняют `config.resolved.yaml` без внутренних классов и `problem.json`
с именами полей/уравнений/коэффициентов, `coefficients.json` с сигнатурами, плюс `log.log`; при включённых метриках — `csv/`.
При ошибке процесс возвращает 1 и записывает traceback в `error.log`, при успехе 0.
functions.py — исполняемый Python-код; изоляцию задачи обеспечивает внешний Docker.

## Переход со старого API

Перенесите x_interval/x_points в `domain.space.x`, t_interval/t_points в
`domain.time`. Вместо net.layers задайте только network.hidden_layers.
Удалите train_datasets, model, callbacks, trainer и все `_target_`.
Перенесите epochs/steps/batch_size/seed в training, устройство — в runtime,
границы — в boundaries. Вместо pde_fn/initial_conditions/output_fn экспортируйте
Problem с equations/initial/transform. Примеры в examples уже используют новый API.
Старые checkpoints могут потребовать нового обучения, если прежняя архитектура
или служебное состояние не соответствует текущей строгой загрузке.

### Переключаемые веса loss

```yaml
training:
  loss:
    kind: fixed       # equal: прежние веса 1/1/1
    initial: 1
    equations: 1
    boundaries: 3
    gradient_norm_every: 200  # 0 выключает диагностику
```

Веса конечные и положительные. kind: equal допускает только единичные веса.
train/terms/initial, equations, boundaries — невзвешенные компоненты;
train/loss — сумма с выбранными весами. train/gradient_norms/* — нормы градиентов
невзвешенных компонентов по обучаемым параметрам, полученные без изменения
optimizer .grad. Диагностика требует дополнительных autograd проходов; её
стоимость входит во время обучения. Recipe сохраняется в config.resolved.yaml,
checkpoint hyper_parameters.loss_settings и benchmark training_recipe.
При predict веса loss не изменяют forward; при продолжении обучения используется
явно заданный текущий recipe. Это fine-tune весов, а не автоматический resume
optimizer state. Старые checkpoints без recipe трактуются как equal.

### Снижение learning rate

```yaml
training:
  steps: 6000
  optimizer:
    learning_rate: 0.001
  schedule:
    kind: exponential   # constant по умолчанию
    final_ratio: 0.1
```

Scheduler обновляется после каждого optimizer step, а не после эпохи.
После 6000 обновлений LR равен 0.0001; последний training step использует
предыдущее значение. Начальная скорость берётся из optimizer.learning_rate.
Exponential требует явного steps; constant не меняет прежнее поведение.
train/learning_rate показывает эффективное значение. Recipe/version сохраняются
в hyper_parameters.rate_settings, scheduler state — в checkpoint Lightning.
Состояния optimizer/scheduler можно восстановить совместно; файловый runner
--checkpoint остаётся fine-tune весов, а не продолжением optimizer state.
Benchmark: --lr-final-ratio 0.1 включает decay, 1 отключает; --steps 12000
даёт отдельный контроль увеличения бюджета с constant LR.

### Обновление PDE-точек

```yaml
training:
  seed: 1234
  collocation:
    sampling: lhs
    points: 1024
    resample_every_epochs: 1  # 0: фиксированная прежняя выборка
```

Обновляются только collocation координаты, IC/BC и holdout не меняются.
Первая выборка прежняя pyDOE LHS; последующие используют независимый NumPy
Generator и Latin hypercube strata, seed + generation. Глобальный RNG не меняется.
Epoch определяется CombinedLoader max_size_cycle, не только PDE-loader.
Refresh выполняется до создания/предзагрузки epoch iterator. Поэтому включается
reload_dataloaders_every_n_epochs=1; фиксированный режим этого не делает.
В состоянии datamodule checkpoint сохраняются текущие точки, seed/частота и epoch;
восстановление sampling state проверяется отдельно. CLI --checkpoint остаётся
fine-tune, не полным resume optimizer/dataloader position. Validation не может
обновлять точки. Benchmark flags: --resample-every-epochs 1 и
--collocation-points 4096 для раздельного контроля более плотной fixed выборки.

### Точные IC и Дирихле на прямоугольнике

training.constraints: hard_initial_dirichlet включает преобразование
u = g(x) + tau * B(x) * N(x,t), где g задано Problem.initial, tau — нормированное
время, B — произведение 4*xi*(1-xi) для нормированных пространственных координат.
Это использует только начальное условие, не аналитическое решение во времени.
IC и BC исключаются из train loss: остаётся PDE. Их качество всё равно оценивается.

Поддержаны полные Dirichlet faces для всех fields: константы или
`value_function` из `Problem.boundary_values`, включая зависимость от времени.
Для нулевых стенок используется прежний fast path. Для остальных автоматически
строится lifting A граничных данных и применяется
`u = g + A(x,t) - A(x,t0) + tau*B*N`.

IC должно совпадать с BC при t0; значения соседних граней должны совпадать на
их пересечениях во все времена. Проверка выполняется на конфигурационной
сетке, не доказывает тождества между узлами произвольного callback. Neumann/Robin,
неполные/повторные грани, variable parameters и custom Problem.transform
отклоняются. Машинный остаток FP32 допустим; это не BC penalty.

Checkpoint фиксирует constraint kind, SHA функций и ненулевые boundary specifications: predict/fine-tune с другим
преобразованием отклоняется. Soft по умолчанию сохраняет прежний forward.
Hard transform не требует autograd при predict, но сохраняет производные при train.
Benchmark flag: --constraints hard_initial_dirichlet.

CUDA runner записывает training.performance.json с peak_allocated_bytes и
peak_reserved_bytes PyTorch allocator, от сброса статистики перед fit до его конца.
Это не вся память GPU/драйвера. Benchmark переносит эти поля в performance.
Для старых checkpoint без профиля поле отсутствует — память не считается измеренной.

### Adam → L-BFGS

```yaml
training:
  steps: 3000                  # Adam updates, unchanged meaning
  optimizer: {learning_rate: 0.001, weight_decay: 0}
  refinement:
    steps: 100                 # additional L-BFGS optimizer calls; 0 disables
    learning_rate: 1.0
    history_size: 50
    tolerance_grad: 1.0e-7
    tolerance_change: 1.0e-9
```

После Adam запускается отдельная полноразмерная стадия L-BFGS с strong Wolfe
line search и max_iter=1 на внешний шаг. Один такой шаг может вычислять closure
несколько раз; при сходимости некоторые вызовы не изменяют веса.
Все текущие IC/BC/PDE-координаты фиксируются на всю стадию, включая случай,
когда во время Adam был включён resampling. Начальное условие и граничные
операторы не заменяются: полноразмерный objective сохраняет их нормализацию.
Используется полный batch каждой группы, поэтому память может заметно вырасти.

Поддерживаются 32-true и один device, без weight_decay. Стадия совместима с soft
и применимыми hard constraints. Диагностика градиентов отдельных компонентов
в closure не запускается. Аналитические ошибки не используются для line search
или остановки. Финальные веса отражают обе стадии.

checkpoints/adam.ckpt сохраняет окончание Adam; best/last относятся к Adam,
final.ckpt — к полному запуску. global_step остаётся числом Adam-шагов.
refinement_state содержит optimizer/history, фиксированные batches, version,
completed и счётчики стадии. hyper_parameters.refinement_settings фиксирует recipe.
При сохранённом model state отдельная стадия умеет восстанавливать состояние;
CLI --checkpoint по-прежнему fine-tune, не автоматическое продолжение Adam/L-BFGS.
Для полного Lightning resume финального staged checkpoint нельзя полагаться
только на его Adam optimizer_states: завершённая стадия отмечена отдельно.

training.performance.json и benchmark performance содержат adam_steps,
adam_seconds (fit), adam_model_evaluations, adam_residual_points, lbfgs_seconds,
lbfgs_steps, lbfgs_internal_iterations, lbfgs_closure_evaluations,
lbfgs_condition_evaluations, lbfgs_residual_points. Последний показатель — число
посещений PDE-точек, не количество скалярных уравнений. CUDA peak охватывает
обе стадии. Полное training_seconds бенчмарка дополнительно включает запуск
процесса/подготовку/сохранение: оно не равно только adam_seconds + lbfgs_seconds.

Benchmark flag --lbfgs-steps 100 добавляет стадию к указанному --steps,
--lbfgs-steps 0 оставляет прежний Adam. Для комбинации используется также
--constraints hard_initial_dirichlet. Сравнение по числу внешних шагов не
считается сравнением вычислительных бюджетов: смотрим closure counts, точки,
время, память и независимую точность. Report сохраняет refinement_recipe.

refinement_state.device и report.refinement_device фиксируют фактическое
устройство стадии после fit. Бенчмарк проверяет completed, version, число
вызовов и backend: CPU-refinement не принимается как CUDA-результат.
adam_model_evaluations и total_model_evaluations считают вызовы model_step,
то есть вычисления суммарного objective. Внутри одного такого вызова несколько
условий могут выполнять отдельные forward; lbfgs_condition_evaluations
показывает число вычислений этих групп в стадии.

### Экспериментальный backend hard predict

```yaml
training:
  constraints: hard_initial_dirichlet
  hard_execution: fused_predict  # default: eager
```

`fused_predict` вычисляет каждый пространственный фактор как
`torch.addcmul(xi, xi, xi, value=-1)` вместо `xi * (1 - xi)` только при
выключенных градиентах. В вещественной арифметике это одинаковый полином;
FP32 округления могут отличаться. При обучении и запросе координатных
производных сохраняется прежний eager graph. Режим не меняет веса,
оптимизатор, checkpoint или тип математического ограничения. Старый hard
checkpoint можно загружать с любым из этих двух backend. Для soft выбор
`fused_predict` отклоняется. Это opt-in для измерений; ускорение всего
файлового runner не гарантируется.

Сравнение на одном сохранённом CUDA checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_hard_execution.py --trial numerical-reports/lbfgs-staged-hard-002/heat3d-1234 --report numerical-reports/hard-execution-001.json --cold-runs 3
```

Команду выполнять на Windows после обязательной установки и проверки
рабочей директории из AGENTS.md. Report должен быть новым файлом.

### Производные hard по правилу произведения

```yaml
training:
  constraints: hard_initial_dirichlet
  hard_execution: product_rule  # экспериментальный training backend
```

Пользовательские `e.dt("u")`, `e.diff("u", "x", 2)` и `e.laplacian("u")`
сохраняются. Для `u = g + tau*B*N` backend вычисляет первые и вторые
производные по правилу произведения; производные сети и начального поля
получает через autograd. Производные более высокого порядка и производные
произвольных Tensor-выражений используют обычный autograd физического поля.
Кеш производных живёт только в пределах текущего forward: он не переносится
между batch, resampling, шагами Adam или closures L-BFGS.

В predict без градиентов используется обычное вычисление поля. Это отдельный
режим, его нельзя одновременно выбирать с `fused_predict`. Веса и тип hard
ограничения одинаковы: checkpoint допускает переключение execution backend.
Hyperparameters checkpoint записывают backend обучения. Применимость BC и
параметров остаётся такой же, как у `hard_initial_dirichlet`; подтверждение
производительности проводится на heat, FP32, текущем CUDA-стеке. Default eager
сохраняется до численной проверки длительного обучения.

Benchmark: `--constraints hard_initial_dirichlet --hard-execution product_rule`.
Независимый PDE holdout всегда дифференцирует физическое поле обычным autograd,
обходя training fast path. Аналитическое решение используется только для оценки.

### Внутренний план ограничений

`SolverFactory.constraints` — неизменяемый `ConstraintPlan`, разрешённый один раз
до создания backend. Он содержит `kind`, `exact_initial`, `exact_faces`,
`remaining_losses`, математический `recipe`, callable `output` и checkpoint
`signature`. Soft сообщает отсутствие гарантированных IC/BC и оставляет
initial/equations/boundaries loss. Текущий hard сообщает IC всех полей, заданные
Дирихле на всех гранях и только equations loss.

Гарантия hard математически условна: `g|Γ=h(t0)` и совместимость соседних traces должны выполняться на всей границе.
Текущая проверка совместимости проверяет значения на конфигурационной сетке,
не доказывает тождество произвольной пользовательской функции между узлами.
Углы/рёбра входят в эту проверку объединением масок граней; это не отдельные
штрафы в hard loss. Совместимость производных классического решения в углах
пока не проверяется.

`HardBoxInitial` задаёт eager-рецепт с optional lifting. `ProductRuleHard` и
`FusedPredictHard` содержат этот же рецепт и выбираются в `hard_executor` один
раз. В forward нет выбора по строке конфигурации. Предметная гарантия и
checkpoint signature не зависят от execution adapter: predict/fine-tune могут
переключать eager/product_rule/fused_predict, сохраняя физическую конструкцию.
Optimizer state при этом не перепараметризуется.

Поддержаны ненулевые Dirichlet BC; автоматический heat profile и hard default
пока не введены. PDE не определяется по имени функции.
Неймана/Робина, variable parameters и custom output transform
для текущего hard явно отклоняются, без soft fallback. Для этих задач по-прежнему
можно явно выбрать soft.


Для временных стенок в `config.yaml`:

```yaml
training:
  constraints: hard_initial_dirichlet
  hard_execution: product_rule
boundaries:
  - {kind: dirichlet, field: u, axis: x, value_function: wall}
  - {kind: dirichlet, field: u, axis: y, value_function: wall}
```

В `functions.py` объявите `wall(c: Coordinates) -> torch.Tensor` и передайте
`boundary_values={"wall": wall}` в `Problem`. Callback получает координаты
именно грани: её нормальная координата зафиксирована на lower/upper. Можно
указать разные функции на разных сторонах через `side`; их значения на
пересечениях должны совпадать. Возвращайте Tensor-столбец (или scalar для BC),
сохраняйте autograd: никакого NumPy/detach в differentiable callback.

Готовые manufactured примеры: `benchmarks/problems/heat_dirichlet2d/functions.py`
и `heat_dirichlet3d/functions.py`; CLI `--cases heat_dirichlet2d heat_dirichlet3d
--constraints hard_initial_dirichlet --hard-execution product_rule`.

### Точные Неймана и Робина в 1D–3D

Режим для одного поля на отрезке, прямоугольнике или параллелепипеде:

```yaml
training:
  constraints: hard_initial_robin
  hard_execution: robin_grouped
boundaries:
  - kind: robin
    field: u
    axis: x
    side: both
    alpha: 1.5707963267948966
    beta: 1
    value_function: wall
```

Для Неймана замените `kind` на `neumann` и уберите `alpha`/`beta`.
Можно задавать условия отдельно через `side: lower` и `side: upper`.
`wall(c)` объявляется в `Problem.boundary_values`; получает физическую координату
соответствующей грани, остальные координаты и текущее время. Константа задаётся через `value`.
Нужны условия на всех гранях без повторений. Для 2D добавьте пару условий на `y`,
для 3D — также на `z`; функции разных граней могут различаться.

Условие Робина здесь использует внешнюю нормальную **производную**, а не поток:

$$
\alpha u+\beta\frac{\partial u}{\partial n}=h(t),\qquad
\partial_n u\big|_{x=a}=-u_x(a,t),\quad
\partial_n u\big|_{x=b}=u_x(b,t).
$$

Коэффициенты постоянные, `beta` ненулевой. Начальное поле обязано удовлетворять
тому же граничному оператору при начальном времени: проверяются значения **и
производные**. При несовместимости запуск отклоняется. В обучении остаётся
только невязка PDE, начальные и граничные условия задаются конструкцией.

Поддержаны 1D–3D, одно поле, без варьирующихся входных параметров и пользовательского
преобразования выходов. Чистый Дирихле с `beta: 0`, `heat_flux` и ускорители
`product_rule`/`fused_predict` для этого режима пока отклоняются. На пересечениях
проверяется согласованность **граничных операторов** данных, а не равенство значений
разных граней. Проверки выполняются на сетке конфигурации в float64; они не доказывают
тождество между узлами. Функции должны быть гладкими, поточечными и сохранять
вычислительный граф производных. Не смешивайте строки пакета и не отсоединяйте
координаты внутри начальных или граничных функций. Нормальные производные сети вычисляются и при предсказании;
наружное отключение autograd обрабатывается внутри преобразования.

Определение сохранённой модели учитывает исходные функции, вид условий и
коэффициенты граничных операторов. Пространственный и временной интервалы
при загрузке должны совпадать с обучающей задачей; менять разрешение сетки можно. Вес штрафа не меняет физическую конструкцию.
Примеры: `benchmarks/problems/heat_neumann/functions.py`,
`benchmarks/problems/heat_robin/functions.py`. Математический вывод и численные
результаты — [отчёт](robin-1d-experiments.md).

Многомерные примеры: `benchmarks/problems/heat_neumann2d/functions.py`,
`benchmarks/problems/heat_robin3d/functions.py`. Доказательство совместимости и
вычислительная цена — [многомерная конструкция](robin-box.md). Прямой вариант
вычисляет сеть 9 раз в 2D и 27 раз в 3D; ускорение этим режимом не обещается.

Для экспериментального пакетного исполнения этой же конструкции задайте
`training.hard_execution: robin_batch` или `robin_grouped`. Второй вариант
разделяет пакет по группам граней и считает только нужные нормальные производные. Оно объединяет граничные вычисления сети,
сохраняя смешанные производные и градиенты параметров. Это выбор способа вычисления,
а не изменение граничных условий; сохранённая модель совместима с `eager`.

При `constraints: hard_initial_robin` по умолчанию применяется подтверждённый
групповой адаптер `robin_grouped`. Для контрольного прямого вычисления укажите
`hard_execution: eager`. Выбор самих ограничений по умолчанию остаётся `soft`;
точную конструкцию нужно включить явно в пределах поддержанного класса задач.

### Независимый способ продолжения данных Робина

```yaml
training:
  constraints: hard_initial_robin
  hard_execution: robin_grouped
  hard_lifting: expanded
```

`hard_execution` выбирает расчёт проекции **сети**, `hard_lifting` — продолжения
**заданных граничных данных**. Это независимые способы вычисления одной физической
конструкции. `hard_lifting: recursive` — контроль и пока значение по умолчанию;
`expanded` раскрывает сумму по пересечениям граней и объединяет два времени
в один поточечный пакет. Способ можно менять при загрузке сохранённой модели.
Для `soft` и Дирихле значение `expanded` отклоняется. Область применимости
Неймана/Робина не расширяется. Математика: [robin-lifting-optimization.md](robin-lifting-optimization.md).

В численном бенчмарке есть соответствующий флаг `--hard-lifting expanded`.
Изолированное сравнение одних сохранённых весов:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_robin_lifting.py --trial numerical-reports/EXPERIMENT/heat_neumann3d-19 --report numerical-reports/NEW-REPORT.json --trace
```

Команду CUDA выполняйте только на Windows из обязательной директории проекта,
с guard и настройкой временных путей из `AGENTS.md`. `--trace` включает отдельный
диагностический проход profiler, не добавляемый к измеренному времени.

### Экспериментальная плотность времени PDE-точек

```yaml
training:
  collocation:
    sampling: lhs
    points: 512
    time_power: 1  # default: 0, равномерное время
```

При положительной степени физические времена смещаются к концу интервала.
Это меняет вес невязки PDE, не само уравнение или его производные. Начальные и
граничные точки, сетка предсказания не меняются. Допускается вместе с обновлением
LHS; степень входит в проверку восстановления состояния выборки. Для `grid`
положительная степень отклоняется; validation остаётся равномерной.
Математика и протокол сравнения: [temporal-sampling-experiments.md](temporal-sampling-experiments.md).

### Конечный обратный проход

```yaml
training:
  backward: auto  # по умолчанию; all — явный контроль
```

`parameters` вычисляет конечные градиенты функционала только для обучаемых весов.
Все производные по координатам **внутри PDE** сохраняются. Настройка применяется
и к Adam, и к L-BFGS; допускается одно CPU/CUDA-устройство с `32-true`.
MPS, смешанная точность и несколько устройств пока отклоняются. Если инструменту
нужны `.grad` входных координат после конечного обратного прохода, выбирайте `all`.
Режим сохраняется как `hyper_parameters.backward_mode`, но не меняет физическую
сигнатуру сохранённой модели: при предсказании или новом обучении из весов его
можно переключать. Бенчмарк принимает `--backward auto|all|parameters`; его контроль по умолчанию — `all`.
Математическое обоснование и ограничения: [parameter-backward.md](parameter-backward.md).

По умолчанию `training.backward: auto` использует `parameters`
только для `hard_initial_robin` на одном CPU/CUDA с `32-true`; в остальных
случаях сохраняет `all`. Это ограничивает автоматический выбор проверяемой
областью, а не молча включает ускорение на MPS или смешанной точности.
Явное `parameters` вне поддерживаемого устройства/точности по-прежнему выдаёт
ошибку. В конфиге сохраняется запрошенный режим, в `hyper_parameters.backward_mode`
— фактически выбранный `all` или `parameters`. Решение принято после
24 CUDA-обучений в 12 парах с неизменными порогами: [результаты](backward-study-2026-10-15.md).

Если `runtime.device` задан как `auto`, политика также сохраняет `all`:
для автоматического ускорения явно укажите `cpu` или `cuda`. Явный
`backward: parameters` с устройством `auto` отклоняется до запуска.

В названии `parameters` имеются в виду **обучаемые веса модели**, а не входные
параметры семейства задач вроде теплопроводности κ. Значения таких входов
по-прежнему подаются в сеть и уравнение; оптимизатор их не восстанавливает.

### Выбор фона для точного Неймана/Робина

```yaml
training:
  constraints: hard_initial_robin
  hard_background: auto  # general | static | auto
```

`static` использует начальное поле как фон только при явно числовых значениях
данных на всех гранях. Для любого `value_function` этот режим отклоняется.
`auto` выбирает его для числовых данных, иначе сохраняет общий путь `general`.
Без настройки для `hard_initial_robin` используется `auto`. Для остальных
типов ограничений сохраняется `general`; эта оптимизация к ним не применяется.
Начальная совместимость по-прежнему проверяется. `hard_lifting` независимо задаёт
способ вычисления общего продолжения. Для контрольного эксперимента укажите
`hard_background: general`; значения весов и физический смысл поля сохраняются.
[Обоснование и ограничения](static-robin-background-hypothesis.md).

### Экспериментальные пространственные признаки Неймана

```yaml
network:
  hidden_layers: [32, 32, 32]
  spatial_encoding: neumann_polynomial
training:
  constraints: hard_initial_robin
```

`spatial_encoding` по умолчанию равен `identity`: используется прежняя сеть
с проекцией. Явные варианты `neumann_cosine` и `neumann_polynomial` заменяют
проекцию поправки сети пространственными признаками. Для каждой физической оси
косинусный вариант использует один признак, полиномиальный — два: q и 16p.
Время остаётся самостоятельным входом. Производные PDE вычисляются по полному
графу физических координат, включая смешанные производные по признакам.

Экспериментальные варианты требуют условий Неймана на всех гранях и
`hard_initial_robin`. Значения условий могут быть числовыми или задаваться
функциями: прежний проверенный фон сохраняется. Для Робина эти признаки сами
по себе недостаточны, поэтому такой конфиг отклоняется. Параметры семейства
по-прежнему ограничены контрактом точного режима.

Исполнитель автоматически выбирается `eager`; ранее явно заданный
`hard_execution: robin_grouped` для этих признаков нужно убрать. Здесь нет
скрытой проекции сети и нет объявления скомпилированного исполнения.
`hard_background` и `hard_lifting` сохраняют смысл для фонового поля.

Это другое представление конечной сети, а не прозрачное ускорение прежних
весов. Сохранённую модель нельзя загрузить с другим `spatial_encoding`;
разрешается изменение разрешения сетки при прежних физических интервалах.
Основным остаётся `identity` до отдельной численной проверки.
[Математическое обоснование и ограничения](neumann-polynomial-representation.md).

Для опытного точного упрощения производных **того же** полиномиального
представления можно явно выбрать:

```yaml
network:
  spatial_encoding: neumann_polynomial
training:
  constraints: hard_initial_robin
  hard_execution: neumann_analytic
```

Это правило сокращает обратный вычислительный граф q,P без изменения формул
поля. Доступны только условия Неймана на всех гранях; значения могут быть
числовыми или задаваться функциями. Основной `eager` сохраняется по умолчанию.
Сохранённые веса совместимы между этими двумя исполнителями при неизменных
условиях, области и представлении. Выбранный исполнитель сохраняется в
метаданных; смена `spatial_encoding` по-прежнему меняет физический смысл весов.
Независимый оценщик качества бенчмарка использует обычные производные.
[Вывод правила и ограничения](neumann-analytic-pullback.md).
