# План улучшения точности PINN

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
