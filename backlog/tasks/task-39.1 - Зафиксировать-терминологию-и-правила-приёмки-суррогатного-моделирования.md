---
id: TASK-39.1
title: Зафиксировать терминологию и правила приёмки суррогатного моделирования
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 08:32'
labels: []
dependencies: []
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-research-roadmap.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-terminology.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-validation.md
parent_task_id: TASK-39
priority: high
ordinal: 40000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Нужен общий язык для долгого исследования, чтобы не смешивать гипотезы коэффициента, веса сети, код функции, точность и экономию. Зафиксировать текущий запрос пользователя и этапы без выбора окончательной архитектуры.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Определены гипотеза, контекст задачи, код функции, общий суррогат и независимый эталон; обозначения опубликованы в docs.
- [x] #2 Зафиксированы физические предпосылки оператора и различие температуропроводности и теплопроводности.
- [x] #3 Разделены аналитическая проверка, обобщение по новым гипотезам, стоимость и проверка физических гипотез; этапы связаны с задачами.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Определить цель, язык и физические предпосылки. 2. Записать правила аналитической приёмки и стоимости. 3. Связать этапы с задачами и обновить контекст.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Работа только с документацией и доской; производственный код не менялся, общий набор тестов не повторялся. Для подзадач TASK-39.N краткий research.py пока не поддерживает идентификаторы с точкой; документирован штатный fallback через backlog task view.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Опубликованы цель программы, словарь и обозначения, физические предпосылки, аналитические примеры, независимые уровни проверки и модель окупаемости. Создана родительская TASK-39 с 14 связанными ступенями; ближайшее исследование — TASK-39.2. Проверены ссылки, идентификаторы и математическая запись; формула первого пространственного семейства проверена AD на CPU в FP64 с отрицательным контролем. Обучение не запускалось, архитектура и API не фиксировались.
<!-- SECTION:FINAL_SUMMARY:END -->
