# Development checks

Используйте Python 3.10 и отдельное окружение:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-test-integration.txt
.venv/bin/python scripts/check.py
```

Команда запускает Ruff lint/format, strict mypy и быстрые CPU-тесты, останавливаясь на
первой ошибке. По умолчанию pytest показывает компактный прогресс и итог; при
падении остаются короткий traceback и захваченные сообщения теста. Подробный
режим включает имена тестов, live logs, полный traceback и десять самых медленных
тестов:

```sh
.venv/bin/python scripts/check.py --verbose
```

Для отдельного запуска pytest эквивалентный диагностический режим:

```sh
.venv/bin/python -m pytest -o 'addopts=--strict-markers --color=auto' -o log_cli=true -vv --tb=long --durations=10
```

Проверки не меняют исходники. Для форматирования:

```sh
.venv/bin/python -m ruff format run.py pinnstorch scripts tests/runner tests/numerics tests/integration examples/heat2d examples/heat_parameterized
```

Ruff задаёт стиль и порядок импортов; длина строки 99. Необязательные pre-commit
hooks используют инструменты активированного окружения. Pre-commit устанавливается
отдельно. Dependencies runtime находятся в requirements.txt; Docker использует
Torch 2.1.0/CUDA 11.8, CPU development — ту же версию Torch.

## Как выбирать проверки

`python scripts/check.py` — быстрый набор по умолчанию. Он включает численные
проверки, конфиги, сетки, границы, batching и короткое обучение настоящего Lightning
в текущем процессе. Исключены только тесты с marker `e2e`, которые многократно
запускают файловый runner в отдельных процессах.

```sh
python scripts/check.py --full             # все проверки и тесты
python scripts/check.py --full --verbose   # полный набор с диагностикой
python -m pytest -m e2e                    # только файловый pipeline
python -m pytest tests/integration/test_explicit_boundary.py -m 'not e2e'
```

Обычный `python -m pytest` по-прежнему запускает весь набор. Для небольших изменений
используем быстрые проверки и тесты затронутого компонента. Если меняется файловый
runner, нужно дополнительно выполнить соответствующие e2e-сценарии. Весь набор
запускаем при завершении крупного блока, перед PR/слиянием и при сквозных изменениях
контрактов config/Problem/checkpoint, зависимостей или backend обучения.
Документация/форматирование сами по себе не требуют полного прогона.

`--verbose` управляет подробностью вывода, `--full` — составом тестов; это независимые
флаги. Pre-commit hooks запускают только lint/format/types, без обязательного pytest.

## Types

Точный набор файлов strict mypy перечислен в pyproject.toml. Он включает публичный
Problem API, весь runner, model/data module, samplers, mesh, contracts, сетку,
loader, FCN, вспомогательные численные функции и logger. Это не означает, что
сторонний Lightning/CUDA код полностью статически проверен.

Tensor-аннотации описывают Python-тип, но не форму и устройство. Их проверяет
адаптер и тесты. cast не конвертирует объект; TYPE_CHECKING позволяет импортировать
типы только для анализатора. Подробнее — [архитектура](architecture.md).

## Tests and measurement

Default pytest collection: tests/runner, tests/numerics, tests/integration.
Данные генерируются локально. Проверяются производные, backprop, loss, разные
границы/углы, 1D/2D/3D, параметры, реальные train/retrain/predict процессы,
checkpoints, ошибки, batch-порядок и исполняемый пример API.md.

CLI import и --help проверяются без training dependencies; для остальных тестов
используйте полное integration environment. Numerics можно запустить отдельно
с requirements-test-cpu.txt. Старые upstream-тесты/примеры, рассчитанные на удалённый
Hydra/discrete/inverse pipeline, удалены.

```sh
.venv/bin/python -m scripts.benchmark_equation_api --points 1024 --steps 50
```

Бенчмарк проверяет совпадение невязок/градиента и сравнивает локальную CPU-стоимость
ручного autograd и Equation. Тесты и этот замер не доказывают сходимость физической
задачи, CUDA/MPS корректность или устойчивость mixed precision. Production проверка
на RTX остаётся отдельным этапом.
