"""The existing exact Dirichlet representation preserves nonlinear PDE weight gradients."""

import os
from pathlib import Path

import pytest
import torch

from scripts.audit_nonheat import audit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [19, 73])
def test_burgers_hard_operator_and_every_weight(
    tmp_path: Path, dtype: torch.dtype, seed: int
) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    result = audit(tmp_path / "audit", "burgers", dtype, seed, device, "hard_initial_dirichlet")
    assert result["parameters"] == 2241 and result["state_unchanged"]
    assert result["initial_max"] < 2e-6 and result["boundary_max"] < 2e-6
    assert result["detached_gradient_relative_l2"] > 1e-4
    assert (result["finite_difference_error"] is not None) == (dtype == torch.float64)
