"""Cross-OS pathlib metadata must not prevent loading tensor/optimizer state."""

import os
import pathlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest
import torch

from pinnstorch.runner.checkpoint_io import load_checkpoint


class ForeignPath:
    def __init__(self, windows: bool) -> None:
        self.windows = windows

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        # Pickle can record a foreign constructor without instantiating it here.
        kind = pathlib.WindowsPath if self.windows else pathlib.PosixPath
        value = (
            r"C:\Projects\pinn-runner\checkpoints"
            if self.windows
            else "/projects/pinn-runner/checkpoints"
        )
        return kind, (value,)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("windows", [False, True])
def test_checkpoint_paths_are_portable_without_monkeypatch(
    tmp_path: Path, legacy: bool, windows: bool
) -> None:
    original_windows, original_posix = pathlib.WindowsPath, pathlib.PosixPath
    state = {
        "state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "callbacks": {"dirpath": ForeignPath(windows)},
        "optimizer": {
            "state": {0: {"exp_avg": torch.tensor([0.1, 0.2])}},
            "param_groups": [{"lr": 0.001}],
        },
    }
    path = tmp_path / "state.ckpt"
    torch.save(state, path, _use_new_zipfile_serialization=not legacy)
    if windows != (os.name == "nt"):
        with pytest.raises(NotImplementedError):
            torch.load(path, map_location="cpu")
    restored = load_checkpoint(path)
    expected = (
        PureWindowsPath(r"C:\Projects\pinn-runner\checkpoints")
        if windows
        else PurePosixPath("/projects/pinn-runner/checkpoints")
    )
    assert restored["callbacks"]["dirpath"] == expected
    torch.testing.assert_close(
        restored["state_dict"]["weight"], state["state_dict"]["weight"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored["optimizer"]["state"][0]["exp_avg"],
        state["optimizer"]["state"][0]["exp_avg"],
        rtol=0,
        atol=0,
    )
    assert pathlib.WindowsPath is original_windows and pathlib.PosixPath is original_posix
    assert tmp_path.is_dir()


def test_checkpoint_requires_state_mapping(tmp_path: Path) -> None:
    path = tmp_path / "not-a-state.pt"
    torch.save(torch.ones(1), path)
    with pytest.raises(ValueError, match="state mapping"):
        load_checkpoint(path)
