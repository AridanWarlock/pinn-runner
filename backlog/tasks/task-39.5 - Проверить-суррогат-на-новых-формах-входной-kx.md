---
id: TASK-39.5
title: Проверить суррогат на новых формах входной k(x)
status: In Progress
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 19:23'
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
Исходная и gain8 серии завершены с 0/3 и сохранены отдельно; критерий научного качества открыт. Следующий ограниченный путь: TASK-39.5.1 — контракт приведённого времени и файловый допуск (численный CPU/CUDA-прототип уже проверен); только затем TASK-39.5.2 — отдельный фиксированный опыт качества. Не переходить к k(x,y), не перебирать множители и не ослаблять поздние цели по текущим результатам.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Аналитический эталон и технические допуски приняты: docs/quadratic-shape-reference-2026-10-21.md. Оценка полных кодов зарегистрирована. Исходная cs395q1021 (19e61e3/b5d1cf3) и отдельная gain8 cs398q1021 (011535f/a9cd806) завершены 0/3 каждая, исполнение полное; приёмка качества отрицательная. Худшие случаи включают обучающие углы; PDE и поток улучшились у gain8, позднее поле ухудшилось. Итоги и происхождение: docs/quadratic-shape-quality-result-2026-10-21.md и docs/quadratic-shape-gain8-result-2026-10-21.md. Основной benchmark восстановлен к множителю 1. Время этого блока диагностическое из-за параллельной нагрузки ПК.

Следующие этапы разделены: TASK-39.5.1 — публичный контракт приведённого времени и файловый допуск; затем TASK-39.5.2 — отдельная серия качества. Прототип x,b,s прошёл CPU/CUDA 4/4 без обучения (e50c52e/d063536), все 2273 градиента и конечные разности по полному направлению; публичного контракта и файлового цикла ещё нет. Математика: docs/surrogate-scale-as-clock-2026-10-21.md. Финальный scripts/check.py --full: Ruff/формат/strict mypy, 1124 CPU-теста с e2e. CUDA — отдельные перечисленные в отчёте допуски, не полный аппаратный каталог.

Критерий 3 открыт, цели не ослаблены. docs/quadratic-late-error-counterexample-2026-10-21.md и docs/quadratic-residual-bound-2026-10-21.md объясняют отдельные поздние цели; выборочный PDE RMS не сертификат непрерывной нормы. docs/surrogate-response-conditioning-2026-10-21.md показывает, почему полевая цель 2% не заменяет отклик 5%. Полные выводы и срезы сохранены в документах.
<!-- SECTION:NOTES:END -->
