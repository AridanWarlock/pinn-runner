---
id: TASK-39.4
title: Поддержать целый код входной функции без декартова произведения признаков
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 16:10'
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
- [x] #1 Определены смысл кодирования функции, допустимый класс и ошибка представления; код выбирается как целое и не смешивается между гипотезами.
- [x] #2 Обучение и предсказание не строят произведение всех признаков; проверены согласованные строки, физические производные при фиксированном коде и отрицательные контроли.
- [x] #3 Проверены идентичность сохранённой модели, семантика запросов и файловый цикл, необходимые CPU/CUDA-сценарии; старые контракты не нарушены.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Зафиксировать ограниченный класс кодов и контракт parameter_rows без изменения прежнего режима. 2. Добавить генерацию сетки и LHS по целым строкам, сохранив нормализацию и порядок запросов. 3. Проверить высокую размерность без произведения, физические производные, отрицательные контроли, файловый цикл и идентичность модели. 4. Полный CPU-набор и необходимые CUDA-допуски; обучение новых форм не запускать в рамках этой карточки.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Контракт целых кодов принят: domain.parameter_rows/prediction.parameter_rows; нет произведения признаков, LHS и IC/BC/PDE сохраняют коды. Полный CPU 1066, затем профильные усиленные 4 численных и 2 файловых; данные 19 контролей, включая 24 признака и области 1D/2D/3D. CUDA cs394n1021 4/4 и cs394f1021 2/2, штатные collect/compare=true. Код f71bc18, протоколы 2a09822. Проверены все 2305 градиентов, конечные разности и прежний независимый режим. Это допуск представления/выборки, не качество новых форм. Отчёт: docs/function-code-admission-result-2026-10-21.md. Далее TASK-39.5: аналитический эталон изменяемых форм.
<!-- SECTION:NOTES:END -->
