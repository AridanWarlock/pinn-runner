# TASK-17: Подготовить отдельную среду нового Torch/Lightning

## Цель и исходные требования

Текущий production stack остаётся доступным; Git pin не закрепляет shared venv.

Статус, приоритет и зависимости читайте из короткой карточки, не из истории.

## Критерии результата

- Job выбирает environment id из allowlist и записывает executable/package versions
- Проверены Python/wheels/driver и macOS CPU + Windows CUDA compatibility
- Новая среда не меняет текущую .venv; рассмотрены load weights_only и Lightning loaders

## Связанные материалы

- docs/model-accuracy-plan.md
