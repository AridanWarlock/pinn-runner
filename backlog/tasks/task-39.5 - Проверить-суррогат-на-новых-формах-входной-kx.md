---
id: TASK-39.5
title: Проверить суррогат на новых формах входной k(x)
status: In Progress
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-24 17:47'
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
Ближайший этап — TASK-39.5.4: публичный контракт модального поля, защита базиса/нормировки и файловый допуск, затем самостоятельная зарегистрированная серия качества. Численный CPU/CUDA-прототип уже проверен. Не повторять отрицательные серии, не перебирать множители или бюджеты, не ослаблять поздние нормы и отклики.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Качество свободной FCN, gain8 и приведённого времени завершено тремя отдельными сериями по 0/3. Цели не ослаблены; худшие случаи включают обучающий угол a=0.15,b=1. Итоги: docs/quadratic-shape-quality-result-2026-10-21.md; docs/quadratic-shape-gain8-result-2026-10-21.md; docs/quadratic-clock-quality-result-2026-10-24.md. Время диагностическое, исходный benchmark с множителем 1 сохранён.

TASK-39.5.1 завершена по публичному контракту часов и CPU/CUDA/файловому допуску; TASK-39.5.2 завершена как отрицательное исследование. TASK-39.5.3 завершена: модальный контроль по одним начальным данным прошёл все цели и 128 новых форм, но это не обученная PINN; docs/modal-surrogate-result-2026-10-24.md.

Текущая TASK-39.5.4: модальные временные поправки с точной IC, M=32, 3199 параметров. CPU/CUDA 4/4 прошли без обучения; публичный контракт и качество ещё открыты. docs/modal-initial-lifting-2026-10-24.md. Критерий 3 родителя открыт; перехода к k(x,y) по этим результатам нет.

Уточнение охвата модального контроля: полный прежний набор метрик и 6 откликов выполнен на 12 физических кодах. Дополнительные 128 новых b проверены начальными L2-/энергетическими нормами и формулами переноса, без переподгонки; это не ещё 128 полных прогонов оценщика и не интервальный сертификат.
<!-- SECTION:NOTES:END -->
