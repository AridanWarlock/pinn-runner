# CUDA numerical baseline

`rtx4070ti-standard.json` — реальный успешный прогон commit `47e99ac` на
RTX 4070 Ti, Torch 2.1.0+cu118, Python 3.10.11. Три задачи, два seed, 3000 шагов.
Все шесть отдельных результатов прошли абсолютные пороги приёмки.

`working_tree_dirty: true` связан с незакоммиченным установщиком Python
`python-3.10.11-amd64.exe` в Windows-копии. Перед переключением и запуском
`git diff --quiet` и `git diff --cached --quiet` подтвердили отсутствие изменений
отслеживаемого кода. Установщик не используется бенчмарками.

Полные артефакты находятся на Windows в
`C:\Projects\pinn-runner\numerical-reports\standard-001`.
Git хранит компактный отчёт, а не checkpoints и все MAT-файлы.

Сравнение с зафиксированным baseline:

```powershell
& '.\.venv\Scripts\python.exe' scripts/benchmark_solutions.py --output-dir numerical-reports\comparison-new --baseline benchmarks\baselines\rtx4070ti-standard.json
```

Не заменяйте baseline автоматически после неудачной проверки. Новая физическая
задача/бюджет/контракт требуют отдельного явно проверенного baseline.
