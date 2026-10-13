---
id: TASK-11
title: Добавить стадии Adam → L-BFGS
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 08:37'
updated_date: '2026-10-13 14:20'
labels:
  - adapter
milestone: m-0
dependencies:
  - TASK-3
  - TASK-5
  - TASK-7
documentation:
  - docs/model-accuracy-plan.md
priority: medium
ordinal: 11000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Улучшение оптимизации требует фиксированного objective в closure и честного учёта вычислений.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Есть явное переключение constant Adam / staged optimizer
- [x] #2 Closure не меняет точки между evaluation; сохранены состояния стадий
- [x] #3 Считаются closure/residual evaluations и сравнение при равном wall budget
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Explicit additional full-batch L-BFGS refinement after fixed Adam steps, no default change. Freeze all latest IC/BC/PDE samples for entire refinement; initial implementation single full batch preserves boundary normalization (chunking deferred, measure peak memory). Torch LBFGS max_iter1 strong_wolfe, count closures/points and actual internal iterations. Stage snapshot/optimizer/counters saved, CLI remains weights fine-tune. Reject mixed precision/multi-device/weight decay. CPU mathematical/state/e2e checks and CUDA Adam/staged soft/hard paired runs plus matched wall-time follow-up.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation:separate full-objective stage,max_iter1 strong_wolfe,frozen sample snapshot,freshcoordinateleaves,finiteguards. CPU8unit tests pass including soft/hard derivative objective,bitwise stage-state continuation. New public train/predict e2e2pass; global_step remainsAdam only,adam/final checkpoints explicit. Full check and benchmark CPU smoke ongoing before CUDA.

Full Mac243passed88.55s; CPU benchmark smoke verified hard+refinement public train/final checkpoint/evaluation and counters. CUDA matrix pinnedfb55ee4 starts with stagedhard/stagedsoft,then fresh controls. First heat3d1234 hard3000+100LB:global0.0818%,final0.298%,PDE0.003879;LB1.40s/204closures. No conclusions untilpaired controls/matrix complete.

Critical integration finding:Lightning strategy.teardown moves model toCPU afterfit. Staged001 ran refinement onWindowsCPU,notGPU; exclude from CUDA speed/quality conclusions. Fix explicitmodel.to(strategy.root_device),save actualdevice,add public backendregression test and rerun staged002. FreshAdam controls in matrix remainvalid.

Backend correction checked:full243passed88.90s,strictmypy49files,Ruff; separate3CPU hardware/refinement transfer/publicbackend tests passed. Matrix001 completed; its Adam controls are valid CUDA,staged001 excluded. Rerun real GPU stages on corrected commit, require refinement_device=cuda:0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented opt-in Adam->full-objectiveLBFGS, frozen coordinates/freshleaves, actualbackend+completed guards, Adam/final checkpoints, optimizer/history/snapshot state. CPU244full+3separatebackend tests; CUDA8passed; Windows10stage+18benchmark tests. 34validGPUtrials,102900Adam+2400LBcalls/4728closures. Hard+LB100 improves bothseeds heat3d/heat2d/Burgers; advectionsoft improves. Matched heat3d wall times within0.61%,quality gain persists. Initial12CPUrefinement trials excluded after fixing Lightningteardown. Memory+49%hard;opt-in,no defaults. Evidence docs/lbfgs-experiments-2026-10-13.md andCSV.
<!-- SECTION:FINAL_SUMMARY:END -->
