---
id: TASK-39.12
title: Определить окупаемость суррогата при серии гипотез
status: To Do
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 08:31'
labels: []
dependencies:
  - TASK-39.2
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-research-roadmap.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-terminology.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-validation.md
parent_task_id: TASK-39
priority: high
ordinal: 51000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Смысл программы — экономить полное время серии гипотез, а не только получать маленькое время одного прохода сети. Первую оценку сделать после постоянного k, затем обновлять для принятых классов.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Определены сопоставимые точность, оборудование и состав затрат: эталоны/данные, обучение, подготовка гипотезы, предсказание, проверки и точные пересчёты.
- [ ] #2 По фактическим измерениям рассчитана область окупаемости по числу гипотез; случаи без окупаемости и неопределённость сохранены.
- [ ] #3 Аналитический эталон не объявлен тяжеловесным конкурентом; при сравнении с разностным решателем проверена его точность и повторное использование вычислений.
<!-- AC:END -->
