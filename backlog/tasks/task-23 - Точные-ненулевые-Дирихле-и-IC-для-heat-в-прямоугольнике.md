---
id: TASK-23
title: Точные ненулевые Дирихле и IC для heat в прямоугольнике
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 15:57'
updated_date: '2026-10-13 18:53'
labels:
  - hard-constraints
  - heat
dependencies:
  - TASK-22
documentation:
  - docs/hard-heat-plan.md
priority: high
type: feature
ordinal: 23000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Нулевые Dirichlet грани не покрывают заданную температуру стенок, включая её изменение во времени. Требуются совместимые начальное поле и граничные данные без участия аналитического решения в обучении.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ненулевые временные BC и IC выполняются конструкцией в 1D–3D; несовместимые данные на пересечениях выявляются
- [x] #2 Поле, первые и вторые производные и parameter gradients проверены на смещённых областях, гранях, рёбрах и углах
- [x] #3 Manufactured heat решения подтверждены на CUDA; checkpoint и документация отражают поддержанные условия
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Transfinite lifting A на box: inclusion-exclusion face projections, совместимые traces на пересечениях. u=g+A(t)-A(t0)+tauBN; derivatives полного background через AD. 2. Проверка IC/BC и парных edge/corner traces на config grid/time, явные ошибки, прежний zero fast path. 3. Shifted1D–3D math/grad tests, short manufactured CUDA train/checkpoint/predict; fullCPU один раз.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
b46801c: compiled inclusion-exclusion DirichletLifting, u=g+A(t)-A(t0)+tauBN, nonzero/time-dependent traces, sampled IC/edge compatibility + duplicate errors. Product rule учитывает G_t/G_tt, zerofastpath сохранён. Nonzero checkpoint signature включает sorted boundary specs;test reorder stable/changed compatible walls differ. FullCPU283passed118.89s;после benchmarkmetric/test уточнений fastCPU262passed21e2eexcluded,mypy58,Ruff. Профильныеmath/reference32passed,новыеlifting10passed. Первый full281passed/2failed обнаружил hardcoded Burgers reference и неверные BC coordinates в benchmarktest; исправлено. CUDA12test+2manufacturedtrain запущены, безполнойматрицы.

CUDA b46801c:12pytestpassed27.63s (6CUDA math,4CPUvalidate,2CUDApublicroundtrip);manufactured heat_dirichlet2d/3d seed19 Adam3000+LB100 обаpassed. ModeL2 .02123%/.09402%,PDE .001752/.004594,BC2.44e-7/4.05e-7;train52.33/133.62s,warm52k/152k predict83.93/795.60ms,cold2.450/3.206s. 3Dgenericlifting26terms costly;zero path unchanged. Отчёт docs/nonzero-dirichlet.md и raw JSON;одинseed неmatrix/default confirmation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Точные совместимые nonzero/time-dependent Dirichlet1D–3D через boolean-sum lifting;полный background AD/product rule, sampled IC/edge/corner validation, physical BC checkpoint signature. FullCPU283passed,финальныйfast262/21excluded,mypy58,Ruff;Windows12tests и2manufactured CUDApassed. Общийliftingдороже,оптимизацияотдельно;sampled validation не доказывает arbitrary callback identity.
<!-- SECTION:FINAL_SUMMARY:END -->
