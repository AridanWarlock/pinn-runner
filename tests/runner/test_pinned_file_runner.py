"""Pinned file tests keep the required Windows cwd and explicit child precision."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.integration import test_runner_files as runner


def test_windows_launch_keeps_root_and_selects_pinned_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = Path(r"C:\Projects\pinn-runner")
    snapshot = required / ".gpu-jobs/lifecycle/checkout"
    monkeypatch.setattr(runner, "ROOT", snapshot)
    monkeypatch.setattr(runner.sys, "platform", "win32")
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: required))
    monkeypatch.setenv("PINN_PINNED_REVISION", "a" * 40)
    calls = []

    def execute(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    runner.launch(required / "input", required / "output", "predict", "weights.ckpt")
    command, kwargs = calls[0]
    assert kwargs["cwd"] == required
    assert command[1:3] == ["-B", "-c"]
    assert command[4] == str(snapshot / "run.py")
    assert command[-2:] == ["--checkpoint", "weights.ckpt"]


@pytest.mark.parametrize("bad_cwd", [True, False])
def test_windows_mismatch_stops_before_launch(
    monkeypatch: pytest.MonkeyPatch, bad_cwd: bool
) -> None:
    required = Path(r"C:\Projects\pinn-runner")
    other = Path(r"C:\Windows\System32")
    monkeypatch.setattr(runner, "ROOT", required if bad_cwd else other)
    monkeypatch.setattr(runner.sys, "platform", "win32")
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: other if bad_cwd else required))
    with pytest.raises(RuntimeError, match="guarded project directory"):
        runner.launch(required / "input", required / "output")


def test_real_pinned_bootstrap_argv_import_root_and_precision(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text("""import json,sys,torch
print(json.dumps(dict(argv=sys.argv[1:], root=sys.path[0],
    matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32)))
""")
    command = runner._pinned_command([sys.executable, str(probe), "--sentinel", "a b"])
    cwd = Path(r"C:\Projects\pinn-runner") if sys.platform == "win32" else tmp_path
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data == dict(argv=["--sentinel", "a b"], root=str(tmp_path), matmul=False, cudnn=False)
