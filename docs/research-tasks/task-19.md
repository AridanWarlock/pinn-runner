# TASK-19: Исследовать compile и functional derivative backend

## Цель и исходные требования

Компиляция forward не доказывает поддержку PINN derivative chain. Windows Triton/toolchain — отдельный риск.

Статус, приоритет и зависимости читайте из короткой карточки, не из истории.

## Критерии результата

- Predict и residual-training режимы проверены отдельно на выбранной версии
- Производные, parameter gradients, graph breaks, prep time и recompilations измерены
- Windows и Linux Docker backend различены; непроверенный fallback не называется compiled

## Связанные материалы

- docs/model-accuracy-plan.md
