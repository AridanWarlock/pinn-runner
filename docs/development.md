# Development checks

Используйте Python 3.10 и отдельное окружение:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-test-integration.txt
.venv/bin/python scripts/check.py
```

Команда запускает Ruff lint/format, strict mypy и CPU-тесты, останавливаясь на
первой ошибке. Она не меняет исходники. Для форматирования:

```sh
.venv/bin/python -m ruff format run.py pinnstorch scripts tests/runner tests/numerics tests/integration examples/heat2d examples/heat_parameterized
```

Ruff задаёт стиль и порядок импортов; длина строки 99. Необязательные pre-commit
hooks используют инструменты активированного окружения. Pre-commit устанавливается
отдельно. Dependencies runtime находятся в requirements.txt; Docker использует
Torch 2.1.0/CUDA 11.8, CPU development — ту же версию Torch.

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
