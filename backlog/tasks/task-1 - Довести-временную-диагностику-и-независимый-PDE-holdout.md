---
id: TASK-1
title: Довести временную диагностику и независимый PDE holdout
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 08:46'
labels:
  - diagnostics
milestone: m-0
dependencies: []
references:
  - 'https://github.com/AridanWarlock/pinn-runner/commit/46ce04b'
  - 'http://127.0.0.1:6420/tasks/task-1'
documentation:
  - docs/model-accuracy-plan.md
priority: high
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Общий L2 скрывает ошибки поздних времён и границ. Основные time metrics уже добавлены в 97313b6; требуется завершить диагностику перед экспериментами.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Отчёт содержит PDE residual по фиксированным временным bins и worst-time L2 с корректным нулевым знаменателем
- [x] #2 Границы и interior оцениваются раздельно; диагностика не изменяет параметры модели
- [x] #3 Результаты heat3d-005 воспроизводятся из checkpoint без нового обучения
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Добавить отдельную holdout диагностику с фиксированными временными bins, сохранив прежние 2048 точек и общий PDE RMSE. 2. Добавить worst-time L2 и численные проверки отсутствия мутации весов. 3. Проверить fast и затронутый e2e. 4. Опубликовать код и переоценить heat3d-005 на Windows без обучения; собрать отчёт. 5. Сохранить доказательства и выводы в задаче.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CPU: Ruff/format/strict mypy; 191 fast passed, 17 excluded; 18 targeted passed; 1 benchmark e2e passed. Residual bins reuse identical holdout; model parameters/gradients unchanged in checkpoint test. CUDA reevaluation heat3d-005 pending.

CUDA reevaluation 46ce04b: numerical-reports/diagnostics-001/report.json; original training SHA 389d599. No training; original metrics match heat3d-005 within 1e-12. Worst time = 0.5, L2 12.803/15.808%. PDE RMSE first bin .0609/.0704, final bin .0249/.0263: no comparable late residual blowup. New hypothesis: IC transition and soft boundary/loss balance before causal weighting. Existing quality failure for seed 4321 preserved; diagnostics acceptance succeeds despite model quality failure.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Добавлены fixed holdout temporal PDE bins и worst-time diagnostics. 191 fast CPU tests passed (17 excluded), 18 targeted passed, 1 affected benchmark e2e passed; Ruff/format/strict mypy green. CUDA checkpoint-only reevaluation reproduced old metrics within 1e-12; deterministic diagnostics do not mutate weights or parameter gradients. Code 46ce04b, report numerical-reports/diagnostics-001/report.json.
<!-- SECTION:FINAL_SUMMARY:END -->
