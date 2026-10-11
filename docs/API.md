# API PINN runner

Этот документ описывает файловый интерфейс `run.py`, который запускается как
одноразовый процесс. Сервер и HTTP API не требуются. Документ относится к
поддерживаемому пути FCN, непрерывному времени и прямоугольным областям 1D/2D/3D.
Сниппеты с `executable-*` ниже проверяются настоящим CPU-запуском в тестах.

## 1. Входные файлы и команды

В одной входной директории должны лежать:

- `config.yaml` — описание области, сети, обучения и условий;
- `functions.py` — пользовательские функции Python;
- checkpoint для режимов `predict` и `retrain`.

`data.mat` не читается. Начальные условия задаются функцией, а PDE и границы —
невязками. Эталонное решение для обучения и предсказания не требуется.

```sh
python run.py --mode train --input-dir ./task --output-dir ./result
python run.py --mode predict --input-dir ./task --output-dir ./prediction --checkpoint /absolute/path/result/checkpoints/final.ckpt
python run.py --mode retrain --input-dir ./task --output-dir ./retrained --checkpoint trained.ckpt
```

| Аргумент | Значение |
| --- | --- |
| `--mode` | обязательный: `train`, `predict`, `retrain` |
| `--input-dir` | по умолчанию `/task_data` |
| `--output-dir` | по умолчанию `/task_output` |
| `--checkpoint` | по умолчанию `checkpoint.ckpt` внутри input-dir; относительный путь тоже считается от input-dir |

CLI не принимает Hydra overrides вроде `trainer.accelerator=cpu`: меняйте YAML.
Код выхода 0 означает успех, 1 — ошибку задачи; неверные CLI-аргументы дают 2.
Ошибки после начала выполнения записываются в `error.log`. При отсутствии
обязательного файла причина записывается в лог, но `error.log` может отсутствовать.

Для разработки установите зависимости по [development.md](development.md).
Примеры используют CPU и три шага; они проверяют pipeline, а не сходимость.

## 2. Полный запускаемый пример: 2D и параметр теплопроводности

Сохраните следующий блок как `config.yaml`.

<!-- executable-config -->
```yaml
seed: 1234
save_pred: true
domain:
  x_interval:
  - 0.0
  - 1.0
  t_interval:
  - 0.0
  - 0.5
  x_points: 8
  t_points: 3
  y_interval:
  - 0
  - 1
  y_points: 6
  parameters:
    conductivity:
      interval:
      - 0.1
      - 0.5
      points: 3
train_datasets:
- initial:
    _target_: pinnstorch.data.InitialCondition
    solution:
    - T
- pde:
    _target_: pinnstorch.data.MeshSampler
    collection_points:
    - residual
    use_lhs: false
- boundary:
    _target_: pinnstorch.data.BoundaryConditions
    conditions:
    - kind: dirichlet
      field: T
      axis: x
      side: both
      value: 0.0
    - kind: dirichlet
      field: T
      axis: y
      side: both
      value: 0.0
val_dataset:
- data:
    _target_: pinnstorch.data.MeshSampler
    collection_points:
    - residual
    use_lhs: false
pred_dataset:
- data:
    _target_: pinnstorch.data.MeshSampler
    use_lhs: false
net:
  _target_: pinnstorch.models.FCN
  layers:
  - 4
  - 16
  - 16
  - 1
  output_names:
  - T
model:
  optimizer:
    lr: 0.01
  scheduler: null
  scaler: null
  jit_compile: false
  amp: false
  cudagraph_compile: false
trainer:
  accelerator: cpu
  devices: 1
  max_epochs: 3
  max_steps: 3
  check_val_every_n_epoch: 1
  num_sanity_val_steps: 0
  enable_progress_bar: false
  enable_model_summary: false
  inference_mode: false
callbacks:
  model_summary: null
  rich_progress_bar: null
  model_checkpoint:
    filename: best
data:
  batch_size: 16
prediction:
  parameters:
    conductivity:
    - 0.25
  allow_extrapolation: false
```

Сохраните следующий блок как `functions.py`.

<!-- executable-functions -->
```python
"""Heat equation family on a square, conditioned on conductivity."""

import numpy as np
import torch
from numpy.typing import NDArray

from pinnstorch.utils import gradient


def initial_conditions(
    spatial: NDArray[np.float64], *, parameters: dict[str, NDArray[np.float64]]
) -> dict[str, NDArray[np.float64]]:
    return {"T": np.prod(np.sin(np.pi * spatial), axis=1, keepdims=True)}


def pde_fn(
    outputs: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    *,
    parameters: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    dx, dy, dt = gradient(outputs["T"], [x, y, t])
    outputs["residual"] = dt - parameters["conductivity"] * (
        gradient(dx, x)[0] + gradient(dy, y)[0]
    )
    return outputs
```

В этом примере сеть представляет `T(x, y, conductivity, t)`. Обучение охватывает
коэффициенты от 0.1 до 0.5. Раздел `prediction` используется только в режиме
`predict`, где запрашивается коэффициент 0.25; `train` и `retrain` используют
исходную сетку из `domain`.

### Как читаются служебные ключи YAML

`_target_` — Python-путь класса, который создаёт Hydra. Например,
`pinnstorch.data.InitialCondition` означает импортировать этот класс и передать
ему остальные ключи словаря как аргументы конструктора. `initial`/`pde`/`boundary`
в списке datasets — названия блоков, а не имена специальных Python-функций.
Runner сам добавляет `_partial_`, чтобы сначала создать фабрику dataset, а затем
передать ей общую mesh. Для optimizer это аналогично фабрике, получающей параметры
сети. Вам достаточно указывать содержательные аргументы.

`null` в YAML превращается в None и отключает optional-компоненты, например
scheduler. `[T]` — список, `{batch_size: 16}` — словарь. Размеры задавайте числами,
а не строками. Имена полей/невязок чувствительны к регистру и должны совпадать с
ключами, возвращаемыми Python-функциями.

## 3. Область: domain

| Ключ | Правило |
| --- | --- |
| `x_interval`, `t_interval` | обязательные конечные строго возрастающие пары |
| `x_points`, `t_points` | обязательные целые числа >= 2; bool не принимается |
| `y_interval`, `y_points` | оба вместе включают 2D |
| `z_interval`, `z_points` | оба вместе включают 3D; требуется y |
| `parameters` | необязательный словарь скалярных входных параметров |

`x/y/z_points` задают число узлов на каждой оси, а не суммарное число точек.
Временные границы включены. Начальные условия применяются при `t_interval[0]`,
который не обязан быть нулём.

```yaml
domain:
  x_interval: [0, 1]
  x_points: 20
  y_interval: [-1, 1]
  y_points: 10
  z_interval: [0, 2]
  z_points: 8
  t_interval: [0, 0.5]
  t_points: 6
```

Размер полной пространственно-временной сетки здесь равен `20*10*8*6`.
Параметры дополнительно умножают размер сетки. Сетка сейчас материализуется
в памяти; batching ограничивает размер вычисления сетью, но не устраняет
хранение самой сетки. Начните с небольшой сетки и LHS для PDE.

## 4. Входные параметры и запросы предсказания

```yaml
domain:
  parameters:
    conductivity:
      interval: [0.1, 0.5]
      points: 5
    amplitude:
      interval: [1.0, 2.0]
      points: 3
prediction:
  parameters:
    conductivity: [0.25]
    amplitude: [1.5, 1.0]
  allow_extrapolation: false
```

Это фрагмент: остальные ключи области и сети также необходимы.
Параметры упорядочены как в `domain.parameters`; порядок ключей в запросе не
меняет входы сети. Значения запроса должны быть непустыми списками конечных
чисел без повторений. Допускается один элемент и несортированный список;
указанный порядок значений сохраняется. Неуказанные параметры используют
свою исходную сетку. Для нескольких списков строится декартово произведение.

Имена — ASCII-идентификаторы Python без начального `_`, отличные от x/y/z/t
и имён выходов сети. В обучении каждый параметр требует `interval` и `points >= 2`.
Он является входом, а не обучаемым неизвестным коэффициентом обратной задачи.

Checkpoint сохраняет имена входов и диапазоны нормализации. Запрос одного
коэффициента не меняет нормализацию. По умолчанию запрошенные координаты,
время и параметры должны укладываться в диапазоны checkpoint. Опция
`prediction.allow_extrapolation: true` разрешает выход за них при `predict`;
она не означает, что модель будет точной за обученным диапазоном.
`retrain` не расширяет исходный диапазон нормализации этой опцией.

Скалярные параметры считаются постоянными внутри одной задачи. Поле `c(x,y)`
можно вычислять формулой внутри PDE, но произвольная табличная карта коэффициента
как вход семейства решений пока не поддерживается.

## 5. Сеть: net

```yaml
net:
  _target_: pinnstorch.models.FCN
  layers: [4, 32, 32, 1]
  output_names: [T]
```

Первый размер равен `число пространственных осей + число параметров + 1`.
Последний размер равен числу `output_names`. Между ними находятся скрытые слои.
Все размеры — положительные целые числа; имена выходов уникальны и не должны
совпадать с координатами или параметрами. Каждый выход имеет форму `(N, 1)`.
FCN нормализует входы в [-1, 1] и использует tanh в скрытых слоях.

Для нескольких полей, например `output_names: [u, v, p]`, последний размер — 3.
Функция `output_fn` может преобразовать сырые выходы или получить физические
поля из потенциала; тогда InitialCondition/PDE/границы обращаются к именам
полей, которые возвращает это преобразование.

## 6. functions.py: сигнатуры и формы

| Функция | Обязательность и результат |
| --- | --- |
| `initial_conditions(spatial)` | обязательна; dict[str, NumPy array (N,1)] |
| `pde_fn(outputs, x, ..., t)` | обязательна; словарь тензоров, включая невязки PDE |
| `output_fn(outputs, x, ..., t)` | необязательна; преобразованные поля |
| `boundary_values` | необязательный словарь функций заданных граничных значений |
| `plot_fn(mesh, predictions, train_datasets, val_dataset)` | необязательна; plotting требует собственной реализации |

Если параметры заданы, первые три функции дополнительно принимают
keyword-only `parameters`; даже если функция его не использует, аргумент должен
приниматься. `boundary_values` callbacks тоже получают его. При отсутствии
параметров этот аргумент не передаётся.

`initial_conditions` получает один NumPy-массив `(N, spatial_dimension)`.
Значения `parameters` здесь — NumPy-столбцы `(N, 1)`. Функция возвращает конечные
массивы по именам `InitialCondition.solution`; сетевые производные здесь не нужны.

PDE/output callbacks получают физические координаты отдельными PyTorch-тензорами:

```python
# 1D с параметрами:
def pde_fn(outputs, x, t, *, parameters): ...
# 2D без параметров:
def pde_fn(outputs, x, y, t): ...
# 3D с параметрами:
def pde_fn(outputs, x, y, z, t, *, parameters): ...
```

Это шаблоны сигнатур, не готовые реализации. Координаты, время и параметры —
столбцы `(N, 1)` float32; outputs — dict[str, Tensor]. Для PDE используйте
PyTorch-операции и `pinnstorch.utils.gradient`: преобразование в NumPy/detach
разрывает цепочку производных. Возвращайте невязку, например `T_t - c*laplacian`;
значение правой части уравнения вычитается внутри этой невязки.

`output_fn` вызывается и при предсказании. При её наличии runner разрешает
вычисление производных входов, например для получения скорости из потенциала.
`predict` не вычисляет PDE и начальные условия, но обе функции по-прежнему
обязательны в файле для согласованного описания задачи.

## 7. Обучающие datasets

`train_datasets` — список словарей вида `{имя: конфигурация}`. Имена служат для
чтения конфига. Runner соединяет datasets с одной mesh и пользовательскими функциями.

### Начальное условие

```yaml
- initial:
    _target_: pinnstorch.data.InitialCondition
    solution: [T]
    num_sample: 100
```

Хотя бы одно InitialCondition обязательно. `solution` задаёт контролируемые поля.
Без `num_sample` берутся все пространственно-параметрические точки при начальном
времени. С ним выбирается случайная подвыборка без повторений; число должно
укладываться в размер начальной сетки.

### PDE

```yaml
- pde:
    _target_: pinnstorch.data.MeshSampler
    collection_points: [heat_residual]
    use_lhs: true
    num_sample: 2000
```

Имена `collection_points` должны присутствовать в результате `pde_fn`.
LHS выбирает точки внутри непрерывной области, включая диапазоны параметров.
Выборка создаётся при построении dataset и не генерируется заново на каждом шаге.
Для полной YAML-сетки используйте `use_lhs: false`; в этом режиме `num_sample`
не ограничивает число collocation points. Несколько невязок можно перечислить
в одном списке. Штраф — MSE относительно нуля; runner использует mse, не sse.

## 8. Граничные условия

Все явные условия задавайте внутри одного BoundaryConditions. Его нельзя
смешивать со старым callback API или дублировать несколькими datasets.

```yaml
- boundary:
    _target_: pinnstorch.data.BoundaryConditions
    conditions:
      - kind: dirichlet
        field: T
        axis: x
        side: lower
        value: 100
      - kind: neumann
        field: T
        axis: x
        side: upper
        value: 0
        weight: 2
      - kind: robin
        field: T
        axis: y
        side: both
        alpha: 2
        beta: 3
        value_fn: environment
```

| kind | Невязка |
| --- | --- |
| dirichlet | T - value |
| neumann | dT/dn - value |
| robin | alpha*T + beta*dT/dn - value |
| heat_flux | -conductivity*dT/dn - value |

`axis`: x/y/z, только существующая физическая ось. `side`: lower/upper/both.
Внешняя нормаль отрицательна на lower и положительна на upper.
`weight` по умолчанию 1 и должен быть конечным и положительным.
Для Robin alpha/beta по умолчанию 1; оба нулевыми быть не могут.
Для heat_flux `conductivity` — положительная константа (по умолчанию 1) или имя
положительного параметра, например `conductivity: conductivity`.
Нейман задаёт производную; он не равнозначен тепловому потоку.

`value` по умолчанию 0. Вместо него можно указать `value_fn`, но не оба одновременно:

```python
import torch

def environment(
    spatial: list[torch.Tensor], time: torch.Tensor,
    *, parameters: dict[str, torch.Tensor],
) -> torch.Tensor:
    return 2.0 + 0.0 * time

boundary_values = {"environment": environment}
```

Для задачи без параметров уберите keyword parameters. Здесь spatial — список
физических столбцов, а не NumPy-массив. Функция возвращает конечный скаляр или
столбец `(N,1)`. Значения вычисляются при создании dataset и фиксируются как цели.

### Углы и рёбра

- Дирихле для одного поля объединяется по уникальным точкам. Угол учитывается
  один раз, а разные значения на пересечении вызывают ошибку (atol/rtol 1e-6).
  Проверка действует на узлах сетки, времени и параметров, не доказывая равенство
  произвольных функций между узлами.
- На общей точке применяется максимальный вес граней, а не сумма весов.
- Нейман/Робин/тепловой поток исключают физические рёбра и вершины. При смешанных
  условиях в пересечениях остаётся Дирихле. Границы времени и параметров не
  являются пространственными углами.
- Для производных условий нужны внутренние точки грани: минимум три точки
  на каждой касательной пространственной оси. Пустая выборка вызывает ошибку.
- Разные поля имеют независимые ограничения. Каждая производная спецификация
  даёт отдельный weighted MSE; Дирихле группируется по имени поля.

## 9. Валидация, prediction и batching

```yaml
val_dataset:
  - data:
      _target_: pinnstorch.data.MeshSampler
      collection_points: [residual]
      use_lhs: false
pred_dataset:
  - data:
      _target_: pinnstorch.data.MeshSampler
      use_lhs: false
data:
  batch_size: 128
```

Допускается один validation dataset и один prediction dataset.
Validation измеряет невязку PDE, не ошибку относительно эталонного решения.
Для validation тоже можно использовать LHS с положительным num_sample.
`pred_dataset` обязателен в поддерживаемом runner API; он использует полную
сетку, без `solution` и `collection_points`. Reference datasets здесь не нужны.

`data.batch_size: null` означает один полный пакет. Положительное число включает
пакеты для обучения, валидации и предсказания. Последний неполный пакет сохраняется.
Обучающие выборки перемешиваются; предсказания идут в порядке сетки. При нескольких
обучающих datasets Lightning циклически использует более короткие loaders до
длины самого длинного. Размер и среднее loss конкретного пакета влияют на
оптимизацию; batching не гарантирует ту же траекторию обучения, что полный пакет.

## 10. Оптимизатор, trainer и checkpoints

```yaml
model:
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  scheduler: null
  scaler: null
  jit_compile: false
  amp: false
  cudagraph_compile: false
trainer:
  accelerator: cpu
  devices: 1
  inference_mode: false
  max_epochs: 100
  max_steps: 1000
  check_val_every_n_epoch: 1
```

Runner подставляет Lightning/PINNModule targets и фабрики optimizer/scheduler;
не нужно вручную задавать `_partial_`. `inference_mode` должен быть false,
поскольку PINN использует производные и при validation. Обучение прекращается
при достижении одного из ограничений эпох/шагов. Seed по умолчанию 1234.
Для CUDA измените accelerator на gpu и devices на `[0]`; локальные CPU-проверки
не подтверждают работу CUDA/AMP/JIT. Параметризованные задачи и новый API границ
сейчас требуют JIT=false и непрерывного времени.

Необязательный scheduler задаётся фабрикой PyTorch, например:

```yaml
model:
  scheduler:
    _target_: torch.optim.lr_scheduler.StepLR
    step_size: 50
    gamma: 0.5
```

Он вызывается один раз в эпоху. Для schedulers, использующих monitor
(например ReduceLROnPlateau), текущая реализация использует `val/loss`, поэтому
нужна валидация. Для начала оставляйте `scheduler: null`.

Checkpoint callback по умолчанию следит за val/loss. Если `val_dataset: null`,
задайте `callbacks.model_checkpoint.monitor: train/loss`. `save_pred: true`
сохраняет предсказание после обучения в pickle.

| Файл | Смысл |
| --- | --- |
| `checkpoints/final.ckpt` | всегда сохраняется после успешного fit; последние веса |
| `checkpoints/best.ckpt` | имя задаётся callback; лучшая модель по monitor, если callback успел её сохранить |
| `checkpoints/last.ckpt` | зависит от save_last и вызова checkpoint callback |

При коротком max_steps обучение может завершиться до первой валидации, поэтому
best/last не гарантированы. Предсказание после train/retrain использует лучший
checkpoint, если путь существует, иначе final. Для внешнего predict можно
надёжно указывать final либо реально существующий best.

`retrain` строго загружает веса и проверяет имена/диапазоны входов. Это дообучение
с новым оптимизатором, не восстановление его состояния и счётчика эпох.
Он сохраняет исходные диапазоны нормализации checkpoint. Чтобы обучать новый
диапазон, запускайте отдельное train.

## 11. Результаты и порядок координат

| Артефакт | Когда создаётся |
| --- | --- |
| `log.log` | сообщения runner/train |
| `error.log` | перехваченная ошибка задачи |
| `config.resolved.yaml` | объединённый конфиг с разрешёнными интерполяциями перед созданием модели |
| `csv/**/metrics.csv` | метрики при включённом CSV logger |
| `predictions.pkl` | после train/retrain при save_pred=true |
| `predictions.mat` | predict: координаты, параметры и поля |
| `prediction.json` | predict: имена входов, диапазоны checkpoint/запроса, shape, dtype, batch size |

Для MAT обозначим S число комбинаций пространственных узлов и значений
параметров, Nt число временных точек. x/y/z и каждый параметр имеют форму `(S,1)`;
это согласованные строки координат, а не независимые векторы осей. t имеет
форму `(Nt,1)`. Каждое поле имеет форму `(S,Nt)`.

Порядок: `meshgrid(indexing="ij")`, затем C-order. Последний параметр меняется
быстрее других пространственных/параметрических осей. Во входных пакетах
время меняется быстрее всех. При отсутствии параметров быстрее меняется
последняя физическая ось.

```python
from scipy.io import loadmat

result = loadmat("prediction/predictions.mat")
# Все пространственные строки при коэффициенте 0.25:
mask = result["conductivity"][:, 0] == 0.25
temperature = result["T"][mask, :]
xy = (result["x"][mask, :], result["y"][mask, :])
```

config.resolved.yaml служит для аудита запуска; он содержит конкретные пути
текущего окружения. Для новой переносимой задачи используйте исходный config.yaml.

Сеть вычисляет float32, а координаты сетки создаются как float64. Для десятичных
значений при фильтрации лучше использовать np.isclose. Pickle содержит словарь
тензоров с плоскими строками, без отдельных координат; для обмена результатами
используйте MAT и prediction.json.

## 12. Проверка разработки и ограничения

```sh
python scripts/check.py
```

Команда проверяет Ruff, форматирование, строгие типы в мигрированных модулях
и поддерживаемые CPU-тесты. Полный пример из этого документа выполняется
интеграционным тестом. Проверки охватывают вычисления и файловый pipeline;
точность решения и достаточное обучение нужно оценивать отдельно.

Сейчас нет произвольной геометрии, импорта полей коэффициента из таблиц,
обратных задач в этом API и графиков сравнения с эталоном. Старые библиотечные
примеры и raw boundary callbacks не являются спецификацией runner. Дополнительные
заметки: [task-inputs.md](task-inputs.md), [development.md](development.md).
