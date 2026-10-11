# Проверки решателя

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-test-integration.txt
.venv/bin/python scripts/check.py
```

Все поддерживаемые тесты используют CPU и локальные данные. Numerics проверяет
аналитические производные, высшие производные и обучение FCN. Integration
использует настоящие samplers, модель и Lightning, запускает файловый runner
отдельными процессами, проверяет конфиги, 1D/2D/3D, параметры, границы, углы,
checkpoints, вывод MAT и примеры документации. Mocks контролируют число вызовов
autograd, CLI dispatch и отсутствие вызовов уравнений/initial во время inference.

Для только численных тестов достаточно requirements-test-cpu.txt:

```sh
.venv/bin/python -m pytest tests/numerics
```

CLI и функции уравнений не используют data.mat. Начальные условия задаёт
Problem.initial в functions.py. Подробнее — [API](../docs/API.md).
Короткие CPU-обучения проверяют работоспособность pipeline, не сходимость решения.
CUDA, MPS и mixed precision этим набором не проверяются.
