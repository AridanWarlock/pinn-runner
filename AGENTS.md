# Проверки разработки

- Для небольших изменений запускайте `python scripts/check.py`: Ruff, format,
  strict mypy и быстрые CPU-тесты без marker `e2e`.
- Дополнительно запускайте профильные тесты изменённого компонента. При изменениях
  файлового runner выполняйте затронутые e2e-сценарии.
- Полный `python scripts/check.py --full` нужен при завершении крупного блока,
  перед PR/слиянием, при сквозных изменениях контрактов config/Problem/checkpoint,
  зависимостей или backend обучения. Не повторяйте его на каждом мелком коммите
  и после чисто документальных/форматирующих изменений.
- `--verbose` включает диагностику независимо от выбора быстрого/полного набора.
- В отчёте называйте выполненный набор; исключённые тесты не считаются пройденными.

Подробности и команды: docs/development.md.

# Windows GPU Testing Environment

GPU tests must run on the remote Windows machine, never locally on macOS.

## SSH connection and mandatory working directory

- Connect using `ssh aridanPC`.
- The only permitted Windows working directory is `C:\Projects\pinn-runner`.
- Immediately after connecting, change to this directory with PowerShell
  `Set-Location -LiteralPath 'C:\Projects\pinn-runner' -ErrorAction Stop`.
- Verify `(Get-Location).Path` before executing any project command; stop on a
  mismatch or failed directory change.
- Never leave this directory during the SSH session. Never execute project
  commands from `C:\Windows\System32`, the user's home, or any other directory.
- Never modify, create, or delete files outside `C:\Projects\pinn-runner`.
- All GPU tests, Git operations, builds, and temporary outputs must stay inside
  this directory. Configure caches and temporary output paths accordingly.
- Use PowerShell on Windows, not Unix shell commands.

## Execution pattern

Prefer non-interactive SSH calls to `powershell.exe -NoProfile -NonInteractive`.
The PowerShell command must begin with this directory guard:

```powershell
Set-Location -LiteralPath 'C:\Projects\pinn-runner' -ErrorAction Stop
if ((Get-Location).Path -ne 'C:\Projects\pinn-runner') { exit 1 }
```

Then update and test, checking each exit code:

```powershell
git pull --ff-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# Run the appropriate GPU test command here, with outputs inside the project.
exit $LASTEXITCODE
```

Use correct local-shell quoting or PowerShell `-EncodedCommand` (UTF-16LE) so the
local shell cannot expand PowerShell variables or change the directory guard.
For a read-only connection check, run `git status` after the guard without pulling.

## Workflow

1. Complete the required changes on macOS.
2. Ensure the tested changes are available in the GitHub repository.
3. Connect to `aridanPC` and immediately set and verify the mandatory directory.
4. Pull the latest changes with `git pull --ff-only`; stop if it fails.
5. Run GPU tests from the mandatory directory.
6. Collect results, errors, and logs, then close the SSH connection.
7. Report the results to the main Codex session and continue working on macOS.

Never attempt to run Windows GPU tests on macOS or use a different Windows
working directory.
