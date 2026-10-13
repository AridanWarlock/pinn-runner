# План улучшения точности PINN

Статус на 2026-10-13: реализованы переключаемые loss weights, step-based
exponential LR, обновляемый LHS и hard IC + homogeneous box Dirichlet;
сетки/бюджеты/quality policies бенчмарков разделены. Проведено 36 численных
обучений на текущем CUDA-стеке. Результаты и ограничения:
[accuracy-experiments-2026-10-13.md](accuracy-experiments-2026-10-13.md).
Hard heat3d улучшился на всех пяти заранее выбранных seed; defaults пока
не меняем. Реальные поддержанные YAML-ключи перечислены в [API.md](API.md):
общая структура `training.strategies` ниже остаётся проектом будущего API.

Продолжение: реализована полноразмерная стадия Adam → L-BFGS,
проверена комбинация с hard, четыре PDE и сопоставимый wall-time на heat3d.
Результаты: [lbfgs-experiments-2026-10-13.md](lbfgs-experiments-2026-10-13.md).
Task-11 завершён; общий этап подтверждения defaults остаётся Task-14.

Ветка: codex/model-accuracy. Базовый контроль: codex/gpu-jobs / 389d599,
heat3d-005 (6000 Adam steps, seeds 1234/4321). Ошибка общего L2 4,68/5,76%,
последнего слоя 12,80/15,81%. Пороги не ослабляем. Прежние результаты не
становятся эталоном автоматически.

## Что уже выяснили

Нет скачка на последнем слое; сдвиг эталона на ±один шаг ухудшает результат.
Решение затухает до 0,228 начальной амплитуды, абсолютная ошибка почти постоянна.
На последнем времени boundary RMSE 0,0135–0,0172, interior RMSE 0,0068–0,0079.
Амплитуда основной синусоидальной моды ошибается лишь на 1,7–3,2%: существенно
искажение пространственной формы. Поэтому начинаем с границ и оптимизации,
а не считаем нарушение причинности доказанным.

## Архитектура подключаемых стратегий

Не вводить общий adapter с десятками optional методов. Ввести отдельные typed
Protocol по обязанностям и явную композицию в TrainingPlan, который SolverFactory
компилирует один раз. В горячем пути — tensor operations без выбора строк.

- LossPolicy: получает именованные loss IC/BC/PDE, возвращает взвешенную сумму.
  EqualWeights, FixedWeights, GradientBalancedWeights, CausalWeights.
- SamplingPolicy: создаёт/обновляет только collocation inputs на границе эпох.
  FixedLHS, ResampledLHS, ResidualAdaptive. Seed и счётчик обновлений сохраняются.
- OptimizationPolicy: строит optimizer/scheduler и стадии обучения.
  ConstantAdam, DecayingAdam, AdamThenLBFGS. LBFGS closure использует фиксированные
  точки и полный batch внутри одной стадии; учитывать число closure evaluations.
- ConstraintPolicy: строит преобразование network outputs и сообщает, какие
  ограничения уже выполнены точно и могут быть исключены из loss.
  SoftConstraints, HardBoxDirichlet, HardInitialAndBoxDirichlet.
- TemporalPolicy: WholeInterval, CausalSegments, TimeWindows. Окна хранят состояния
  и проверяют непрерывность; предсказание предыдущего окна не является точным IC.

User YAML: training.strategies с понятными именами и параметрами; без import path
и target. Например learning_rate: {kind: exponential, final_ratio: 0.1},
loss: {kind: fixed, initial: 1, boundaries: 10, equations: 1},
sampling: {kind: resampled, every_epochs: 1},
constraints: {kind: soft}, temporal: {kind: whole_interval}.
Это проект API, пока не поддерживаемые настройки: строгий parser их отклоняет.

Эксперимент задаёт complete resolved strategy set. Для удобства profile: standard
выбирает только подтверждённые defaults; profile: control воспроизводит прежнюю
модель. Каждая стратегия может явно отключаться через базовый kind. Сохранять
resolved settings и strategy version в checkpoint/report, проверять совместимость
при predict/resume. Неприменимую hard constraint отклонять, не игнорировать молча.
Профиль standard версионировать; никакого скрытого изменения старых запусков.

## Порядок реализации

1. Полиморфные BenchmarkCase: reference, boundaries, boundary_mask. Отдельно
   BenchmarkGrid/Budget/QualityPolicy вместо выбора по имени PDE. Сохранить
   текущие конфиги и численные лимиты, проверять их снимками.
2. Временные метрики RMSE/L2/max отдельно для boundary/interior, reference norm.
   PDE residual по временным bins на фиксированном независимом holdout, отдельные
   train loss и периодические gradient norms. Сбор диагностик выключаемый,
   overhead измерять отдельно. Нулевой reference norm => null, не epsilon-обман.
3. Именованные loss terms и LossPolicy; переключаемые BC weights 1/3/10.
4. Constant/decaying Adam и SamplingPolicy fixed/resampled. Каждое изменение
   сначала отдельно; увеличение числа точек — отдельный контрольный эксперимент.
5. Hard constraints для совместимых homogeneous box Dirichlet + initial.
   u = g + (t-t0) B N, где B ноль на гранях. Использовать лишь известные IC/BC,
   не аналитическое решение. Проверить производные, углы, совместимость с
   пользовательским transform; не распространять автоматически на Neumann/Robin.
6. AdamThenLBFGS, затем при необходимости gradient balancing и causal segments.
   Time windows вводить последними: более сложный execution/checkpoint контракт.
7. Матрица абляций, подтверждение defaults, документация API, Windows CUDA tests.

## Численная оценка и правила принятия

- Эталонные PDE: heat2d, heat3d, advection, forced Burgers. Все аналитические
  значения только для оценки, никогда не подаются в train loss.
- Скрининг: прежние два seed, одна стратегия за запуск, одинаковая architecture,
  evaluation grid, holdout, точность и бюджет. Называть это предварительным
  результатом, не статистическим доказательством.
- Подтверждение: пять заранее выбранных seed 1234/4321/19/2026/73, в том числе
  новые, попарное сравнение с контролем на каждой PDE. Report: median/worst,
  каждый seed, разброс; никаких выборов удачного checkpoint по analytic error.
- Метрики: общий и final L2, worst-time L2, time-resolved RMSE/max, interior и BC,
  independent PDE residual, IC error. Не менять grid ради выигрышной оценки.
- Скорость: wall train seconds, optimizer steps, residual/closure evaluations,
  warm inference median, полный predict, peak CUDA memory. Для LBFGS сравнивать
  также при равном времени, шаг Adam не равен шагу LBFGS.
- Предварительный критерий полезности: median final L2 heat3d меньше минимум на
  25%; worst seed не ухудшается; абсолютные/interior ошибки тоже не растут.
  Подтверждение: выигрыш на >=4 из 5 heat3d seed; все quality gates и отсутствие
  регрессий остальных PDE по заранее заданным лимитам. Это инженерное правило,
  не оценка статистической значимости. Лимиты фиксируем до запусков.
- Цена более 25% wall time/memory требует отдельного решения: полезную, но
  дорогую стратегию оставить opt-in. Неприменимые стратегии не делать глобальными.
- После индивидуальных тестов: совместный набор и leave-one-out абляции,
  проверка взаимодействий. В standard входят только стратегии с выигрышем
  в итоговой композиции; до этого standard сохраняет прежнее поведение.

## Новые гипотезы из результатов

Если BC weighting улучшает boundary, но ухудшает PDE/interior — проверять
gradient conflict/balancing, а не бесконечно повышать вес. Если hard BC убирает
основной дефект — проблема ограничений подтверждается. Если resampling уменьшает
holdout residual при похожем train residual — вероятно переобучение точкам.
Если decay помогает, но больше fixed-LR шагов нет — вероятен optimization floor.
Если ошибка растёт уже в абсолютной метрике при качественных IC/BC — исследовать
causality, временные residual bins и окна. Если ни одна стратегия не помогает,
проверить вместимость сети и precision отдельными контролируемыми опытами.

Каждый опыт хранит гипотезу, SHA, resolved config, seeds, изменённый фактор,
метрики, цену, вывод и следующий опыт. Ошибочный запуск не численная неудача.
При возобновлении baseline сравнения различать fingerprint PDE/evaluation,
training recipe и implementation revision: менять стратегии разрешено, менять
задачу/сетку/бюджет незаметно — нет. Старый строгий fingerprint оставить для
повторного использования checkpoint; не ослаблять его для этого.

## Проверки

CPU: строгий parser/отключение стратегий, analytic residuals и derivatives,
loss weighting, воспроизводимость sampling, scheduler boundaries, hard IC/BC,
checkpoint/resume и независимость диагностики от параметров модели.
Windows: CUDA derivatives и train/predict для адаптеров, затем long matrix.
Fast check после отдельных блоков; full при изменении config/Problem/checkpoint
и завершении крупного блока. GPU-запуск pin SHA; работать на Mac параллельно.

Источники:
- https://arxiv.org/abs/2001.04536 — градиенты компонентов loss.
- https://arxiv.org/abs/2207.10289 — sampling/resampling.
- https://proceedings.mlr.press/v235/rathore24a.html — Adam + L-BFGS.
- https://arxiv.org/abs/2203.07404 — causal training.
- https://arxiv.org/abs/physics/9705023 — точные ограничения.

## Обновление библиотек и ускорение backend (исследовано 2026-10-13)

Сейчас runtime закреплён: torch 2.1.0 / CUDA 11.8, lightning и pytorch-lightning
2.1.0, torchmetrics 1.2.0, numpy 1.26.4, scipy 1.11.4, pyDOE 0.3.8.
Официальный релиз PyTorch 2.14 вышел 2 сентября 2026; последний опубликованный
Lightning release — 2.6.6. Это кандидаты для отдельной среды, не уже проверенный
новый production stack. Совместимость wheels/Python/CUDA/драйвера и Lightning
проверить разрешением зависимостей и реальным запуском перед фиксацией lock.

PyTorch 2.14 улучшает eager dispatch/autograd и compiler overhead, вводит
NVGEMM в Inductor. Большая часть других улучшений (attention, distributed/NCCL,
низкая точность, новые GPU) не относится к нашей небольшой Tanh MLP на одной
4070 Ti. MPS улучшения потенциально полезны на Mac, но текущие CPU-тесты они
не ускоряют; MPS не подменяет CUDA acceptance. CUDA Graphs, torch.compile,
fused Adam и torch.func появились раньше 2.14: новый релиз не обязателен для
каждого из этих экспериментов.

### Приоритеты и математический контракт

1. Обновлённый eager FP32: одинаковые уравнения, производные, IC/BC, sampling,
   optimizer hyperparameters. Только затем включать ускорители по одному.
   Прирост от версии неизвестен; возможны нулевой эффект и регрессия.
2. OptimizerExecution: default / foreach / fused Adam. Сохраняет формулы Adam,
   но порядок операций и округления может отличаться. Это не новая стратегия
   обучения: не смешивать с decay или LBFGS при первом сравнении. Проверять
   beta/epsilon/weight_decay/state resume, поддержку CUDA и dtype. На CPU
   eager/default; явно запрошенный неподдерживаемый режим => понятная ошибка.
3. GraphExecution: eager / CUDA graph. При статических tensor shapes уменьшает
   CPU kernel launch overhead. Сначала чистый predict; затем отдельно training
   residual+backward+optimizer. Постоянные input buffers, копирование новых
   значений, warmup и capture. Последний неполный batch обрабатывать отдельным
   путём; padding не должен менять среднее loss или число обучающих примеров.
   Sampling/resampling обновляет данные buffers, не делает точки постоянными.
4. CompileExecution: off / predict / residual_training. Сначала predict без
   производных. PINN training требует производных по координатам, вторых
   производных и затем градиента loss по весам: compiled forward сам по себе
   не доказывает поддержку этого пути. AOTAutograd исторически имел ограничения
   double backward; актуальную поддержку конкретного пути проверять минимальным
   тестом на выбранной версии, не считать исправленной по номеру релиза.
5. DerivativeBackend: текущий autograd / functional torch.func (functional_call,
   vmap, jacrev/jacfwd). Сравнить diagonal Hessian/направленные производные, не
   строить полный Hessian по всей пачке и не включать parameter inputs в Laplacian.
   Разные схемы AD могут быть быстрее или медленнее; проверять градиенты по весам,
   коэффициентам, multi-output и transforms. Это потенциально более полезное
   изменение, чем простое обновление пакета, но существенно сложнее.
6. MathPrecision: strict FP32 / TF32 / mixed. Последние меняют численную точность,
   поэтому отдельная гипотеза, а не прозрачное ускорение. FP16/BF16 могут ухудшать
   вторые производные и small residual; в default только после numerical gates.
   Сохранить dtype и матмул-настройки в отчёте, сравнивать с FP64 derivative checks.

Эти execution adapters ортогональны TrainingPlan из раздела выше. Не использовать
один переключатель «fast», незаметно включающий новую точность/optimizer/sampling.
Каждый report содержит requested/effective execution mode, graph breaks,
число recompilations, preparation seconds и причины отказа. Нельзя обозначать
fallback eager как успешно проверенную compilation.

### Windows, контейнер и совместимость

Официальный tutorial Inductor Windows описывает CPU/XPU, а не NVIDIA CUDA.
Для Windows CUDA существует triton-lang/triton-windows с собственной матрицей
PyTorch/Triton/toolchain. Не считать pip install нового torch достаточным для
GPU compile. Первый этап на Windows — eager/fused/CUDA graphs; compile проверять
также в целевом Linux CUDA Docker. WSL не требуется включать в текущий план.

Новые CUDA wheels используют новые runtime; это отдельное от установленного
Toolkit. До выбора cu126/cu130/etc проверить NVIDIA driver, wheels выбранного
релиза и RTX 4070 Ti; не обновлять driver автоматически. CUDA toolkit/compiler
может понадобиться для JIT. На Mac нужны совместимые CPU/arm64 wheels.

Создать отдельную .venv-next внутри Windows project; текущую .venv не менять.
GPU job должен уметь явно выбирать environment id из allowlist и фиксировать
Python executable + package versions; сейчас он всегда использует .venv, поэтому
такую возможность надо реализовать до сравнительных запусков. Все пути/caches
остаются внутри C:\Projects\pinn-runner. Отдельный Docker image для нового stack.

Миграционные проверки:
- torch.load с 2.6 изменил default weights_only. Централизовать checkpoint I/O,
  явно выбрать режим для нашего формата, тестировать старый checkpoint/new runtime
  и resume optimizer/scheduler. Не исправлять глобальной env-переменной молча.
- torch.compile wrapper может менять state_dict keys: сохранять canonical eager
  model state, проверять eager/compiled roundtrip.
- Новые Lightning versions: dictionary loaders, max_size_cycle/restarts, batch
  counts, max_steps, callbacks, precision/inference_mode и gradient transforms.
  Это способно изменить train semantics даже при неизменном PDE.
- Обновлять согласованно lightning/pytorch-lightning/torchmetrics; проверить,
  нужен ли второй distribution напрямую, прежде чем удалять зависимость.
- NumPy/SciPy обновлять отдельным шагом, учитывать NumPy 2 ABI и сериализацию MAT.
  Замена pyDOE на scipy.stats.qmc — отдельная sampling migration: может менять
  последовательность точек при том же seed, не скрывать её внутри torch upgrade.

### Как оценивать выигрыш

Не обещать проценты из LLM/CNN benchmarks. Для нашей MLP 32x3 исходный heat3d:
53,77 с / 6000 шагов (~8,96 мс/step с включённым runner overhead), warm predict
152303 точек 20–21 мс, полный predict 2,32–2,36 с. Измерить profiler вне итогового
timing: сколько занимают Python/launch, derivatives/backward, optimizer и I/O.

Численные сценарии ниже — расчёт чувствительности, не измеренный прогноз:
- Если optimizer = 10% времени и fused ускорит его вдвое, весь запуск станет
  быстрее всего в 1/0,95 = 1,053 раза (~5% экономии времени).
- Если устранимый launch overhead = 40% и graphs уберут половину, ускорение
  1/0,8 = 1,25 раза (20% экономии времени). Сначала проверить эту долю profiler.
- Ускорение warm predict вдвое сэкономит примерно 10 мс на полном predict 2,34 с:
  меньше 0,5%, если остальной overhead не меняется. Cold compile может ухудшить.
- Если compile preparation = 10 с, а обучение 54 с станет 36 с, полный запуск
  46 с: экономия только ~15%, а не 33%. Для 20-секундного обучения при тех же
  пропорциях compile вообще не окупится. Кеш контейнера по умолчанию холодный.

Matrix: old eager -> new eager -> fused -> graphs -> compile -> derivative backend,
каждый фактор отдельно; precision отдельно. Два seed для screening, пять для
подтверждения, все PDE, одинаковые бюджеты и fresh-process wall timing, >=3
повторов timing. Отдельно warm kernel benchmark без файлов/transfer и cold
end-to-end с подготовкой/compile/checkpoint/MAT. Сравнивать также time-to-quality
по заранее заданным gates, точное совпадение траекторий optimizer не требовать.

Критерий для performance default: повторяемая >=10% экономия end-to-end времени
на значимых задачах без numerical regression и чрезмерного роста памяти;
иначе оставить opt-in. Обновление ради совместимости/исправлений допустимо без
прироста, но называть это maintenance, не ускорением. Dependencies меняются
только после full CPU, CUDA derivatives/train/predict/resume и Docker acceptance.

Источники обновления:
- https://pytorch.org/blog/pytorch-2-14-release-blog/
- https://github.com/Lightning-AI/pytorch-lightning/releases
- https://pytorch.org/get-started/locally/
- https://pytorch.org/get-started/previous-versions/
- https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html
- https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- https://docs.pytorch.org/tutorials/unstable/inductor_windows.html
- https://github.com/triton-lang/triton-windows
- https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html
- https://docs.pytorch.org/docs/main/notes/serialization.html
