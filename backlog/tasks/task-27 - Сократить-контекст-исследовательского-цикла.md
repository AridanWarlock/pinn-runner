---
id: TASK-27
title: Сократить контекст исследовательского цикла
status: Done
assignee:
  - '@codex'
created_date: '2026-10-17 08:25'
updated_date: '2026-10-17 08:34'
labels: []
dependencies: []
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/research-workflow-api.md
priority: high
type: feature
ordinal: 27000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Проверяемый компактный сбор и сравнение результатов без изменения обучения. Требования: docs/research-workflow-api.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Полнота, сопоставимость и математические критерии проверяются; ошибки не дают ложного успеха.
- [x] #2 Краткое чтение Backlog и передача без истории по умолчанию, открытые задачи ссылаются на требования.
- [x] #3 Полная математика сохранена; ограничения и проверки реализации отражены в документации.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Контракт и инварианты. 2. Сбор/сравнение и краткое чтение задач. 3. Проверки и реальные данные. 4. Передача и требования следующего этапа.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Код88c8370, codex/research-tools. Итог: docs/research-tools-result-2026-10-17.md. Следующий блок TASK-28; математическая работа отдельно TASK-21.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Реализован компактный CLI без Torch/обучения; проверены16 завершённых CUDA-результатов и реальный SSH-статус. Полный719, финальный быстрый672 (50e2e исключены), профиль28. Требования и передача: docs/agent-start.md; подробный итог docs/research-tools-result-2026-10-17.md.
<!-- SECTION:FINAL_SUMMARY:END -->
