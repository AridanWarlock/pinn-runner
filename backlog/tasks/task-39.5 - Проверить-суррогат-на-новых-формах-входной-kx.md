---
id: TASK-39.5
title: Проверить суррогат на новых формах входной k(x)
status: In Progress
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 16:28'
labels: []
dependencies:
  - TASK-39.4
references:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/quadratic-shape-reference-2026-10-21.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/quadratic-shape-next-experiment-2026-10-21.md
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-research-roadmap.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-terminology.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-validation.md
parent_task_id: TASK-39
priority: high
ordinal: 44000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Следующий научный шаг после масштаба: менять форму пространственного коэффициента, сохраняя одну модель. Подготовка TASK-38 полезна, но численный эталон её двухпараметрического опыта не заменяет требуемую пользователем аналитическую приёмку.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 До обучения выбрано положительное семейство с независимым изменением формы; аналитические решения или аналитические ряды с контролем остатка проверены.
- [x] #2 Разделены новые координаты и новые функции k; источник и остальные данные фиксированы либо явно включены в полный контекст гипотезы. Нечувствительное к k поле не засчитывается как приёмка суррогата.
- [ ] #3 Общие веса проверены на новых функциях с абсолютными целями и ошибкой кода; сохранены худшие случаи, стоимость и границы переноса.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Первый этап: выбрать аналитически разрешимое семейство с независимой формой k(x), вывести ряд и оценку хвоста; реализовать независимый эталон и CPU-проверки. Качественное обучение запускать отдельным заранее заданным протоколом после проверки эталона.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Подготовлен аналитический ряд для k=a(1+x/b)^2 при фиксированных IC=sin^3(pi*x), BC=0, f=0. Независимая квадратура и 25 CPU-гипотез, доказанные границы хвоста поля и производных: docs/quadratic-shape-reference-2026-10-21.md. CUDA cs395n1021: 4/4, collect/compare=true, код b3a95bf, протокол 5bae460; обучения нет. Следующий протокол: docs/quadratic-shape-next-experiment-2026-10-21.md — 16 обучающих кодов, восемь новых форм и четыре угла, три модели, отдельные нормы поля/производной/потока и отклики по обоим компонентам. Качество общих весов ещё не проверено, критерий 3 открыт.

Файловый CUDA-допуск cs395f1021 завершён 1/1, collect/compare=true: код 2efbb76, протокол 5133204. Только 3 Adam + 2 внешних L-BFGS и точное восстановление, не качество. Полный CPU 1093; после дополнений быстрый 1022 (73 e2e исключены), новый файловый 1/1 отдельно. Следующий узкий блок — зарегистрировать оценку полных пар (a,b), производной/потока и шести откликов; только затем серия качества.
<!-- SECTION:NOTES:END -->
