---
id: TASK-39.2
title: Проверить один суррогат для постоянного коэффициента температуропроводности
status: To Do
assignee: []
created_date: '2026-10-21 08:27'
updated_date: '2026-10-21 08:31'
labels: []
dependencies:
  - TASK-39.1
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-research-roadmap.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-terminology.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/surrogate-validation.md
parent_task_id: TASK-39
priority: high
ordinal: 41000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Самый простой опыт должен проверить саму схему: одна модель получает значение k и предсказывает решения для новых k без нового обучения. Использовать аналитическое семейство с одинаковыми начальными и граничными данными.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 До обучения опубликованы аналитическое семейство, диапазон k, разбиение гипотез, бюджеты, абсолютные цели и критерий окончания.
- [ ] #2 Одна общая модель оценена без переобучения на заранее выделенных новых k; проверены поле, невязка, начальные и граничные данные, сохранены все исходы.
- [ ] #3 Есть несколько начальных состояний сети, контроль игнорирования k, проверка сохранённой модели и исходное измерение полной стоимости; принято решение о следующем этапе.
<!-- AC:END -->
