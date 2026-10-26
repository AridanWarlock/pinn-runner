---
id: TASK-39.5
title: Проверить суррогат на новых формах входной k(x)
status: In Progress
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-26 09:57'
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
TASK-39.5.4 завершила публичный FP32-контракт, защищённую нормировку и интегральный PDE-функционал с техническими CUDA-допусками. Следующий научный шаг — одна заранее зарегистрированная серия трёх общих моделей, 14 обучающих форм и независимые 12 кодов/6 откликов. Протокол modal-fp32-quality-protocol.json, source b7bd415, protocol cbbbe68. Критерий 3 остаётся открытым до полных результатов качества; к k(x,y) не переходить.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Критерии 1/2 выполнены, критерий 3 открыт: обученные общие веса на новых формах пока научно не приняты. Свободная FCN, gain8 и приведённое время дали по 0/3; отчёты quadratic-shape-quality-result-2026-10-21.md, quadratic-shape-gain8-result-2026-10-21.md, quadratic-clock-quality-result-2026-10-24.md сохранены. TASK-39.5.1–3 завершены; идеальный модальный контроль не является обученной PINN. 26 октября TASK-39.5.4 завершила публичный FP32-контракт и интегральный функционал: CUDA cs395po1026 8/8, cs395ph1026 1/1, cs395tf1026 1/1, collect/compare=true; полный CPU 1230 с e2e, Ruff/формат/mypy прошли. Подробности: docs/modal-admission-result-2026-10-26.md. Одна серия качества зарегистрирована, но не запускалась: source b7bd415, protocol cbbbe68, modal-fp32-quality-protocol.json, 3 seed и 500+500. Прежние пороги/12 кодов/6 откликов сохранены. До результатов не переходить к k(x,y); стоимость только диагностическая.
<!-- SECTION:NOTES:END -->
