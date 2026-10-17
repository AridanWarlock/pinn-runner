# TASK-15: Профилировать текущий CUDA backend и проверить fused Adam

## Цель и исходные требования

Ускорение не требует сначала обновлять Torch. Нужно найти долю kernel launch, derivatives и optimizer.

Статус, приоритет и зависимости читайте из короткой карточки, не из истории.

## Критерии результата

- Profiler собирается отдельно от итогового timing
- Default/foreach/fused сравниваются при неизменных hyperparameters и FP32
- Измерены cold wall time, warm step, память, numerical gates и resume

## Связанные материалы

- docs/model-accuracy-plan.md

Прежний план и полная история: `docs/research-history/task-15.md`. Открывать только для
проверки происхождения конкретного вывода; это не действующий план.
