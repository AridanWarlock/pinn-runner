---
id: TASK-25
title: Удешевить разность Dirichlet lifting без изменения PDE
status: Done
assignee:
  - '@codex'
created_date: '2026-10-13 19:15'
updated_date: '2026-10-13 19:24'
labels: []
dependencies:
  - TASK-23
ordinal: 25000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Общий lifting3D требует двух проходов по26слагаемым. Пользователь разрешил объединить проходы и переиспользовать геометрию; сохранить IC/BC,physical derivatives иparameter gradients без нового API или detached cache.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Values/first-second derivatives/PDE/parameter gradients совпадают с прежними двумя проходами на shifted1D–3D FP32/float64,IC/faces/edges/corners
- [x] #2 CUDA timing на одинаковых weights/points сравнивает baseline и fused difference; проверены checkpoint/public train/refine/predict/fine-tune, без полной long matrix
- [x] #3 Документация и benchmark отражают measured speed/limitations; inline math заменена отображаемыми блоками
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. One-forward prefix projection/weight cache и difference term sum; legacy callable остаётся numericalcontrol/fallback. 2. Reference math tests, CPUfull одинраз, short CUDApair timing, publicroundtrip. 3. Зафиксировать измерения, поправить LaTeXmarkup безinline $.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
50db44c: one-forward prefix geometrycache и termwise difference; старый __call__ numericalcontrol, customcallablefallback. Никакогоdetach;callbackcount52 в3D сохраняется. Reference field/deriv/grad tests baselineoldbackground. FullCPU283passed113.89s,mypy58,Ruff;explicit benchmarkmypy1passed;lifting10passed. Inline math заменена plainvariable names+displayblocks,правилоAGENTS. CUDA12tests+samecheckpoint timing128/2048/17 underway; no long trainmatrix.

CUDA50db44c:12tests27.11s; matchedcheckpoint residual+backward14–15% faster2D/23%3D,forward28–29%/36%,128/2048/17 и reversedorder. Max field2.861e-6,residual2.384e-7,paramgrad6.194e-8. Public reeval existing ckpts passed without training: warm2D83.93→60.23ms/3D795.6→509.46ms,mode .0212285%/.0940245%,physicalPDE .001752/.004594. Coldsingle2.450→2.414/3.206→2.887s неrobust pairedconfidence. Rawreports docs/experiments/2026-10-13-lifting-{shared,predict}.json; reportdocs/lifting-optimization.md. No full train/time-to-quality claims.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Shared geometry/time difference сохраняетmathematical hardrecipe иgraph gradients;baselinecallable retained. FullCPU283passed,mypy58,Ruff+explicitscriptmypy;CUDA12tests passed,matchedcheckpoint timing выигрыш14–23% residual/backward и28–36%predict. Publicwholegrid3D795.6→509.46ms,qualitypassed,no newlongtraining. Inline math исправлена,AGENTSpreference сохранена;limitations/rawreports recorded.
<!-- SECTION:FINAL_SUMMARY:END -->
