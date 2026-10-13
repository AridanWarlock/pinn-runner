---
id: TASK-22
title: Выделить контракт hard constraints для heat-профиля
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-13 15:56'
labels:
  - hard-constraints
  - heat
dependencies: []
documentation:
  - docs/hard-heat-plan.md
priority: high
type: enhancement
ordinal: 22000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Существующий HardBoxInitial смешивает математическую конструкцию и исполнение. Расширение heat на другие BC требует явных гарантий, применимости и checkpoint-контракта, сохраняя прямой hard и soft как отдельные режимы.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Разрешённая стратегия сообщает точные IC/BC и оставшиеся loss; unsupported hard отклоняется без молчаливого soft fallback
- [ ] #2 Eager и opt-in execution отделены от математического рецепта; горячий путь не выбирает стратегии по строкам
- [ ] #3 Save/load/predict/fine-tune и прежний поддержанный hard проверены; CPU full и профильные CUDA проверки задокументированы
<!-- AC:END -->
