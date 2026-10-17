---
id: TASK-17
title: Подготовить отдельную среду нового Torch/Lightning
status: Deferred
assignee: []
created_date: '2026-10-13 08:37'
updated_date: '2026-10-17 08:34'
labels:
  - runtime
milestone: m-1
dependencies:
  - TASK-14
  - TASK-15
  - TASK-16
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-tasks/task-17.md
priority: low
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Требования и критерии: docs/research-tasks/task-17.md. Сначала прочитать этот документ; подробную историю загружать только для конкретного вопроса.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Job выбирает environment id из allowlist и записывает executable/package versions
- [ ] #2 Проверены Python/wheels/driver и macOS CPU + Windows CUDA compatibility
- [ ] #3 Новая среда не меняет текущую .venv; рассмотрены load weights_only и Lightning loaders
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Подробности в документе требований; новые отчёты добавлять ссылками.
<!-- SECTION:NOTES:END -->
