---
id: TASK-39.5
title: Проверить суррогат на новых формах входной k(x)
status: To Do
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 08:31'
labels: []
dependencies:
  - TASK-39.4
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
- [ ] #1 До обучения выбрано положительное семейство с независимым изменением формы; аналитические решения или аналитические ряды с контролем остатка проверены.
- [ ] #2 Разделены новые координаты и новые функции k; источник и остальные данные фиксированы либо явно включены в полный контекст гипотезы. Нечувствительное к k поле не засчитывается как приёмка суррогата.
- [ ] #3 Общие веса проверены на новых функциях с абсолютными целями и ошибкой кода; сохранены худшие случаи, стоимость и границы переноса.
<!-- AC:END -->
