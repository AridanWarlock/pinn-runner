# TASK-16: Проверить CUDA Graphs на текущем стеке

## Цель и исходные требования

Маленькая MLP может ограничиваться launch overhead. Статические buffers не должны замораживать sampling.

Статус, приоритет и зависимости читайте из короткой карточки, не из истории.

## Критерии результата

- Eager/graph predict сравниваются отдельно; cold capture включён в wall time
- Если training поддерживается, проверены вторые производные и parameter gradients
- Неполный batch и обновление sampling сохраняют численный смысл; effective mode отражён в отчёте

## Связанные материалы

- docs/model-accuracy-plan.md
