---
id: TASK-3
title: Ввести именованные loss и отключаемый LossPolicy
status: To Do
assignee: []
created_date: '2026-10-13 08:37'
labels:
  - adapter
milestone: m-0
dependencies: []
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
IC/BC/PDE сейчас складываются без индивидуальных весов. Нужна основа для контролируемых опытов, не изменение defaults.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 EqualWeights воспроизводит текущий loss; FixedWeights задаётся строгим YAML
- [ ] #2 Регистрируются IC/BC/PDE losses и периодические gradient norms без изменения обучения
- [ ] #3 Resolved strategy set сохраняется в отчёте и checkpoint; отключение возвращает контроль
<!-- AC:END -->
