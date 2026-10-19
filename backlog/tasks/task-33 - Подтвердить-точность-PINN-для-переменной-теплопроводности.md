---
id: TASK-33
title: Подтвердить точность PINN для переменной теплопроводности
status: Done
assignee:
  - '@codex'
created_date: '2026-10-18 11:30'
updated_date: '2026-10-19 18:44'
labels:
  - accuracy
  - physics
dependencies:
  - TASK-21
documentation:
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/variable-conductivity-experiment-draft-2026-10-18.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/variable-conductivity-network-result-2026-10-19.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/variable-conductivity-cuda-result-2026-10-19.md
  - >-
    https://github.com/AridanWarlock/pinn-runner/blob/codex/research-tools/docs/variable-conductivity-convergence-result-2026-10-19.md
priority: low
type: spike
ordinal: 33000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Пользователь просит вернуться к расширению физического охвата. Коэффициенты формулами/таблицами и Equation.divergence уже есть; нужна проверенная точность решения. Первый охват — заданный гладкий положительный скалярный коэффициент, зависящий от координат; не обратная задача и не разрывные материалы.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Задана консервативная PDE и совместимые начальные/граничные данные; независимая проверка обнаруживает ошибочную замену дивергенции потока на коэффициент, умноженный на лапласиан.
- [x] #2 Проверены физическая невязка и градиенты по весам для формулы и таблицы; описаны пределы интерполяции и гладкости.
- [x] #3 До CUDA-обучения опубликован протокол с фиксированными бюджетами, абсолютными порогами и несколькими начальными состояниями; отчёт содержит качество, время и неудачи.
- [x] #4 Проверены файловые train/predict/retrain и защита от подмены коэффициента; обратная задача не объявлена решённой.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Завершены постановка, CPU48 всех весов, CUDA kappa02016/16 без обучения,5 файловых CPU-сценариев и пределы интерполяции. 2. Существующий контракт2 расширен физической идентичностью, представлением коэффициента, SHA параметров отдельно от буферов и метрикой потока; прежние протоколы сохранены. 3. Длинный kappa021 отменён и сохранён отдельно. Две перспективные стадии500+500: kappa0228/8 и kappa02312/12;20 уникальных исходов на5 состояниях,все исходные абсолютные пороги,оба collect/compare=true. 4. Аудит происхождения/бюджетов/полноты выполнен,итоговый отчёт8a992ce опубликован,критерии4/4. Основной режим и зависимости не менялись.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
19.10: Done4/4 в охвате известной аффинной kappa в2D. CPU48/CUDA kappa02016 и5 файловых CPU-сценариев подтверждают оператор/градиенты/контракт. Сходимость:20/20 уникальных финальных исходов,5 состояний,формула/таблица,500+500,кодfbcc9a2;пакеты kappa022/023 и протоколы439b1ec/b27e874,оба collect/compare=true. kappa0212000+2000 сохранён отменённым и исключён;14/20 имеют соседнее подтверждение. Полный отчёт и ограничения: docs/variable-conductivity-convergence-result-2026-10-19.md,коммит8a992ce. Не обратная задача и не гарантия произвольных материалов;defaults прежние.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Подтверждена точность выбранной2D консервативной задачи с известной аффинной kappa:20/20 финальных исходов на5 состояниях,формула/таблица,500 Adam+500 L-BFGS;два отдельных перспективных collect/compare=true. CPU48/CUDA16 проверяют оператор и все весовые градиенты,5 файловых CPU-сценариев — жизненный цикл/подмену материала. Отчёт docs/variable-conductivity-convergence-result-2026-10-19.md,кодfbcc9a2,результат8a992ce. Отмена длинной попытки,промежуточные неудачи и пределы вывода сохранены;defaults не менялись.
<!-- SECTION:FINAL_SUMMARY:END -->
