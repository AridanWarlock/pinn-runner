---
id: TASK-39.4
title: Поддержать целый код входной функции без декартова произведения признаков
status: To Do
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 12:24'
labels: []
dependencies:
  - TASK-39.3
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-research-roadmap.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-terminology.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-validation.md
parent_task_id: TASK-39
priority: high
ordinal: 43000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Для расширения от масштаба к форме материала недостаточно добавлять независимые параметры. TASK-38 обнаружила экспоненциальную сетку и LHS, разрушающий коррелированные коды. Нужен ограниченный проверяемый контракт целой функции, без преждевременной фиксации общей архитектуры.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Определены смысл кодирования функции, допустимый класс и ошибка представления; код выбирается как целое и не смешивается между гипотезами.
- [ ] #2 Обучение и предсказание не строят произведение всех признаков; проверены согласованные строки, физические производные при фиксированном коде и отрицательные контроли.
- [ ] #3 Проверены идентичность сохранённой модели, семантика запросов и файловый цикл, необходимые CPU/CUDA-сценарии; старые контракты не нарушены.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Архитектурное ограничение будущего кода функции: docs/surrogate-function-input.md. Два положительных гладких профиля с одинаковым локальным k(x0) дают разные начальные скорости температуры; девять CPU-контролей AD подтвердили формулу до 1.12e-16. Требуется описание всей гипотезы, а не только локальное значение. Это подготовка требований, реализация TASK-39.4 не начиналась.
<!-- SECTION:NOTES:END -->
