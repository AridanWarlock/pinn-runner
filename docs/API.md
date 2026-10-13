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
