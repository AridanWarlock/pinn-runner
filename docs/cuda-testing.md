# Проверки Windows + CUDA

## Подготовка (PowerShell из корня репозитория)

Используем Python 3.10 x64 и текущие зависимости проекта. Вначале получаем ту же
ветку и commit, что проверяли на Mac. Отдельное окружение не затрагивает ваши
другие проекты:

```powershell
py -3.10 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
& $python -m pip install -r requirements-test-integration.txt
& $python -m pip check
& $python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

Индекс CUDA 11.8 для Torch 2.1.0 указан в
[официальных командах PyTorch](https://pytorch.org/get-started/previous-versions/).
Остальные зависимости ставим из обычного индекса. Если is_available=False, до
обучения выясняем причину: драйвер, доступность GPU или установленная сборка Torch.
Запишите вывод nvidia-smi вместе с отчётом: он содержит версию драйвера. CUDA version
в nvidia-smi и torch.version.cuda описывают разные вещи: возможности драйвера и
runtime сборки Torch. Устанавливать весь CUDA toolkit ради этих тестов не требуется.

## Запуск

```powershell
& $python scripts/check.py --full
& $python scripts/check_backend.py --output-dir backend-report
$LASTEXITCODE
```

Код 0 — все acceptance-тесты прошли, 1 — падение pytest, 2 — CUDA недоступна
(тесты не запускались). Другие коды также означают неуспех. Скрипт не подменяет CUDA
на CPU автоматически. Сохраните environment.json, tests.log, results.xml и вывод
nvidia-smi. Используйте новый output-dir для каждого прогона, чтобы не смешать
артефакты разных запусков. Git SHA в отчёте предполагает чистую рабочую копию;
перед запуском проверьте `git status --short`.

На Mac тот же маршрут можно проверить так:

```sh
python scripts/check_backend.py --device cpu --output-dir /tmp/pinn-backend-cpu
```

Это проверка инструмента и CPU-маршрута, а не проверка CUDA.

## Что проверяем

- В 1D/2D/3D оператор div(k grad T), производные интерполируемой таблицы k,
  loss.backward через производные поля: аналитическое значение и CPU/GPU.
- Файловое обучение 3D с параметром, LHS, таблицей коэффициента и mini-batches;
  границы Дирихле, Неймана и теплового потока.
- Предсказание из CPU-checkpoint на GPU и из GPU-checkpoint на CPU,
  включая перенос зарегистрированных buffers коэффициента.
- Конечные веса/выходы и соответствие предсказаний между устройствами.

Базовый режим — float32 (`32-true`). Сравнение имеет численный допуск, а не
побитовое равенство: [PyTorch объясняет различия вычислений CPU/GPU](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).
Совпадение траекторий отдельных обучений CPU/GPU не требуется: сравниваем
предсказания одних и тех же весов. Эти короткие тесты не доказывают сходимость.
Mixed precision, скорость, долгие запуски и запуск внутри Linux Docker проверим
отдельно после базовой приёмки. Windows Python не заменяет production-контейнер.

GPU-набор находится вне стандартного testpaths, поэтому `check.py --full` не
требует GPU. Он проверяется lint и типами, а выполняется отдельной командой выше.
