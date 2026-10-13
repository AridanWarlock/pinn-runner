"""Closure correctness and refinement history survive a backend transfer."""

import io
import os
from pathlib import Path

import pytest
import torch

from tests.numerics.test_refinement import make_stage


@pytest.mark.parametrize("hard", [False, True])
def test_refinement_checkpoint_transfer_to_cpu(tmp_path: Path, hard: bool) -> None:
    stage = make_stage(tmp_path / "source", hard)
    expected = stage.closure()
    device = torch.device(os.environ.get("PINN_TEST_DEVICE", "cpu"))
    stage.model.to(device)
    stage.objective = stage.objective.to(device)
    torch.testing.assert_close(stage.closure().cpu(), expected, rtol=2e-4, atol=2e-5)
    metrics = stage.run(3)
    stream = io.BytesIO()
    torch.save({"model": stage.model.state_dict(), "stage": stage.state_dict(metrics)}, stream)
    stream.seek(0)
    state = torch.load(stream, map_location="cpu")
    restored = make_stage(tmp_path / "cpu", hard)
    restored.model.load_state_dict(state["model"])
    restored.load_state_dict(state["stage"])
    assert all(clock.device.type == "cpu" for _, clock, _ in restored.objective.batches.values())
    observed = restored.run(2)
    assert observed["lbfgs_steps"] == 5
    assert torch.isfinite(restored.closure())
